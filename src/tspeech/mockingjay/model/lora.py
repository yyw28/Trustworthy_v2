"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

LoRA adds trainable low-rank matrices to existing linear layers,
dramatically reducing the number of trainable parameters while
maintaining fine-tuning effectiveness.
"""
from typing import Optional, Union
import math
import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a Linear layer.
    
    Instead of fine-tuning the full weight matrix W (d x d), LoRA learns
    two smaller matrices A (d x r) and B (r x d) such that:
    W' = W + (B @ A) * (alpha / r)
    
    where r is the rank (typically 8-64) and alpha is a scaling factor.
    
    Parameters
    ----------
    linear_layer : nn.Linear
        The original linear layer to wrap
    rank : int
        Rank of the LoRA matrices (default: 8)
    alpha : float
        Scaling factor for LoRA weights (default: 8.0)
    dropout : float
        Dropout probability for LoRA (default: 0.0)
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Get input and output dimensions
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Initialize LoRA matrices
        # A: initialized with zeros (so initial output is unchanged)
        # B: initialized with Kaiming uniform (small random values)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize B with small random values (using standard Kaiming init)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: original output + LoRA output.
        
        Parameters
        ----------
        x : Tensor
            Input tensor
            
        Returns
        -------
        Tensor
            Output tensor
        """
        # Original output (frozen)
        original_output = self.original_layer(x)
        
        # LoRA output: x @ A^T @ B^T * scaling
        # More efficient: (x @ A^T) @ B^T
        x_dropout = self.dropout(x)
        lora_output = (x_dropout @ self.lora_A.T) @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the original layer weights.
        This is useful for inference to avoid the extra computation.
        """
        with torch.no_grad():
            # W_new = W_old + (B @ A) * scaling
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_W
    
    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"original_shape=({self.original_layer.out_features}, {self.original_layer.in_features})"
        )


def inject_lora_to_linear(
    module: nn.Module,
    linear_name: str,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> LoRALinear:
    """
    Replace a Linear layer with a LoRA-wrapped version.
    
    Parameters
    ----------
    module : nn.Module
        Parent module containing the linear layer
    linear_name : str
        Name of the linear layer attribute (e.g., 'query', 'key', 'value')
    rank : int
        LoRA rank
    alpha : float
        LoRA alpha scaling factor
    dropout : float
        LoRA dropout probability
        
    Returns
    -------
    LoRALinear
        The LoRA-wrapped linear layer
    """
    if not hasattr(module, linear_name):
        raise ValueError(f"Module {module} does not have attribute '{linear_name}'")
    
    original_linear = getattr(module, linear_name)
    if not isinstance(original_linear, nn.Linear):
        raise ValueError(f"Attribute '{linear_name}' is not a Linear layer, got {type(original_linear)}")
    
    # Create LoRA wrapper
    lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha, dropout=dropout)
    
    # Replace the layer
    setattr(module, linear_name, lora_linear)
    
    return lora_linear


def apply_lora_to_transformer_layers(
    model: nn.Module,
    target_layers: Optional[list[int]] = None,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> dict[str, LoRALinear]:
    """
    Apply LoRA to transformer encoder layers in Mockingjay.
    
    Parameters
    ----------
    model : nn.Module
        The Mockingjay model (UpstreamExpert)
    target_layers : list[int] | None
        List of layer indices to apply LoRA to (e.g., [10, 11]).
        If None, applies to all layers.
    rank : int
        LoRA rank (default: 8)
    alpha : float
        LoRA alpha scaling factor (default: 8.0)
    dropout : float
        LoRA dropout probability (default: 0.0)
    target_modules : list[str] | None
        List of module names to apply LoRA to.
        Default: ['query', 'key', 'value', 'dense'] for attention and feed-forward.
        
    Returns
    -------
    dict[str, LoRALinear]
        Dictionary mapping layer names to LoRALinear instances
    """
    if target_modules is None:
        target_modules = ['query', 'key', 'value', 'dense']
    
    # Access transformer encoder
    if not hasattr(model, 'transformer'):
        raise ValueError("Model does not have 'transformer' attribute")
    
    transformer = model.transformer
    if not hasattr(transformer, 'model'):
        raise ValueError("Transformer does not have 'model' attribute")
    
    model_module = transformer.model
    if not hasattr(model_module, 'encoder'):
        raise ValueError("Transformer model does not have 'encoder' attribute")
    
    encoder = model_module.encoder
    if not hasattr(encoder, 'layer'):
        raise ValueError("Encoder does not have 'layer' attribute")
    
    layers = encoder.layer
    
    # First, freeze ALL parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Determine which layers to process
    if target_layers is None:
        target_layers = list(range(len(layers)))
    
    lora_modules = {}
    
    # Apply LoRA to each target layer
    for layer_idx in target_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        
        layer = layers[layer_idx]
        
        # Apply to attention modules
        if hasattr(layer, 'attention'):
            attention = layer.attention
            if hasattr(attention, 'self'):
                self_attn = attention.self
                # Apply to query, key, value
                for module_name in ['query', 'key', 'value']:
                    if module_name in target_modules and hasattr(self_attn, module_name):
                        lora_key = f"layer.{layer_idx}.attention.self.{module_name}"
                        lora_modules[lora_key] = inject_lora_to_linear(
                            self_attn, module_name, rank=rank, alpha=alpha, dropout=dropout
                        )
                
                # Apply to attention output dense
                if hasattr(attention, 'output') and hasattr(attention.output, 'dense'):
                    if 'dense' in target_modules:
                        lora_key = f"layer.{layer_idx}.attention.output.dense"
                        lora_modules[lora_key] = inject_lora_to_linear(
                            attention.output, 'dense', rank=rank, alpha=alpha, dropout=dropout
                        )
        
        # Apply to feed-forward modules
        if hasattr(layer, 'intermediate') and hasattr(layer.intermediate, 'dense'):
            if 'dense' in target_modules:
                lora_key = f"layer.{layer_idx}.intermediate.dense"
                lora_modules[lora_key] = inject_lora_to_linear(
                    layer.intermediate, 'dense', rank=rank, alpha=alpha, dropout=dropout
                )
        
        if hasattr(layer, 'output') and hasattr(layer.output, 'dense'):
            if 'dense' in target_modules:
                lora_key = f"layer.{layer_idx}.output.dense"
                lora_modules[lora_key] = inject_lora_to_linear(
                    layer.output, 'dense', rank=rank, alpha=alpha, dropout=dropout
                )
    
    return lora_modules


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        The model
        
    Returns
    -------
    tuple[int, int]
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
