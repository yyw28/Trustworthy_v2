from typing import Optional
import torch
import torchmetrics
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F
from s3prl import hub

from tspeech.mockingjay.model.lora import (
    apply_lora_to_transformer_layers,
    count_trainable_parameters,
)


class HTModel(pl.LightningModule):
    def __init__(
        self,
        mockingjay_config_path: str,
        trainable_layers: int,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_target_layers: Optional[list[int]] = None,
        lora_target_modules: Optional[list[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load Mockingjay model from s3prl
        # Use hub.mockingjay() to load the pretrained model
        # If mockingjay_config_path is just "mockingjay", use default
        # Otherwise it can be a specific variant like "mockingjay_960hr"
        if mockingjay_config_path == "mockingjay":
            self.mockingjay = hub.mockingjay()
        else:
            # Try to get the specific variant from hub
            if hasattr(hub, mockingjay_config_path):
                self.mockingjay = getattr(hub, mockingjay_config_path)()
            else:
                # Fallback to default mockingjay
                self.mockingjay = hub.mockingjay()
        
        # Keep Mockingjay on CPU for stability (avoids GPU memory issues)
        # This ensures stable training without GPU memory problems
        self.mockingjay = self.mockingjay.cpu()

        # Apply LoRA adapters for efficient fine-tuning
        if use_lora:
            # Default: apply LoRA to last two layers (10, 11) if not specified
            if lora_target_layers is None:
                # Use trainable_layers to determine which layers to target
                # If trainable_layers=2, target the last 2 layers (10, 11)
                num_layers = 12  # Mockingjay has 12 transformer layers
                lora_target_layers = list(range(num_layers - trainable_layers, num_layers))
            
            # Default target modules: attention (query, key, value) and feed-forward (dense)
            if lora_target_modules is None:
                lora_target_modules = ['query', 'key', 'value', 'dense']
            
            # Apply LoRA to transformer layers
            self.lora_modules = apply_lora_to_transformer_layers(
                self.mockingjay,
                target_layers=lora_target_layers,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            
            # Freeze all original Mockingjay parameters, but keep LoRA parameters trainable
            # LoRA parameters are in LoRALinear modules (lora_A and lora_B)
            for name, param in self.mockingjay.named_parameters():
                # Keep LoRA parameters trainable (they're named 'lora_A' or 'lora_B')
                if 'lora_A' in name or 'lora_B' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Count parameters
            trainable, total = count_trainable_parameters(self.mockingjay)
            print(f"\n{'='*60}")
            print(f"LoRA Configuration:")
            print(f"  Target layers: {lora_target_layers}")
            print(f"  Target modules: {lora_target_modules}")
            print(f"  Rank: {lora_rank}, Alpha: {lora_alpha}")
            print(f"  Trainable parameters: {trainable:,}")
            print(f"  Total parameters: {total:,}")
            print(f"  Trainable ratio: {trainable/total*100:.2f}%")
            print(f"{'='*60}\n")
        else:
            # Legacy: Freeze all Mockingjay parameters except the last two transformer layers
            # Mockingjay has 12 transformer layers (layer.0 through layer.11)
            # We'll unfreeze layers 10 and 11 (the last two layers)
            self.lora_modules = {}
            for name, param in self.mockingjay.named_parameters():
                # Unfreeze the last two transformer layers (layer.10 and layer.11)
                if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Get hidden size from Mockingjay to create classifier layer
        # Purpose: We need to know the feature dimension (768) to create Linear(768 → 1)
        # 
        # Flow: Mockingjay outputs (batch, seq_len, 768) → Pool → (batch, 768) → Classifier → (batch, 1)
        # Without knowing 768, we can't create the classifier!
        #
        # Get hidden_size from the attention layer structure
        try:
            attention = self.mockingjay.transformer.model.encoder.layer[0].attention.self
            hidden_size = attention.all_head_size  # This is 768 for Mockingjay
        except (AttributeError, IndexError):
            # Fallback: Mockingjay always uses 768, but check model structure first for robustness
            hidden_size = 768
        
        self.linear = nn.Sequential(nn.Linear(hidden_size, 1))

        self.f1_val = torchmetrics.F1Score(task="binary")
        self.acc_val = torchmetrics.Accuracy(task="binary")
        self.f1_test = torchmetrics.F1Score(task="binary")

        self.trainable_layers = trainable_layers
        self.use_lora = use_lora
        
        # Note: Mockingjay is kept on CPU via to() override
        # This prevents Lightning from moving it to GPU during setup

    def to(self, *args, **kwargs):
        """
        Override to() to keep Mockingjay on CPU.
        
        PyTorch Lightning calls model.to(device) during setup.
        This override ensures Mockingjay stays on CPU even when
        the rest of the model is moved to GPU/MPS.
        """
        # Call parent to() for all modules except mockingjay
        result = super().to(*args, **kwargs)
        # Ensure Mockingjay stays on CPU
        if hasattr(self, 'mockingjay'):
            self.mockingjay = self.mockingjay.cpu()
        return result

    def forward(self, wav: Optional[Tensor] = None, mask: Optional[Tensor] = None, mel_spectrogram: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass: Audio/spectrogram → Features → Classification
        
        STEP-BY-STEP FLOW:
        ┌─────────────────────────────────────────────────────────────┐
        │ INPUT: wav (batch, time) OR mel_spectrogram (batch, time, mels) │
        └─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ STEP 1: Extract features using Mockingjay transformer      │
        │   - If wav: Convert to unpadded list → Mockingjay → features│
        │   - If mel: Feed directly to transformer → features         │
        │   Output: hidden_states (batch, seq_len, 768)              │
        └─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ STEP 2: Pool sequence to single vector per sample          │
        │   - Apply mask to ignore padding                           │
        │   - Mean pool over time dimension                          │
        │   Output: pooled_features (batch, 768)                     │
        └─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────────────────────────┐
        │ STEP 3: Classify                                           │
        │   - Linear layer: 768 → 1                                  │
        │   Output: logits (batch, 1)                                 │
        └─────────────────────────────────────────────────────────────┘
        
        Parameters
        ----------
        wav : Tensor, optional
            Shape: (batch_size, sequence_length). Waveform data.
        mask : Tensor, optional
            Shape: (batch_size, sequence_length). True=valid, False=padding.
        mel_spectrogram : Tensor, optional
            Shape: (batch_size, time_frames, num_mels). Mel spectrogram.

        Returns
        -------
        Tensor
            Shape: (batch_size, 1). Binary classification logits.
        """
        # Route to appropriate input handler
        if mel_spectrogram is not None:
            return self._forward_from_mel(mel_spectrogram, mask)
        elif wav is not None:
            return self._forward_from_wav(wav, mask)
        else:
            raise ValueError("Either 'wav' or 'mel_spectrogram' must be provided")
    
    def _forward_from_mel(self, mel_spectrogram: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        STEP 1: Extract features from mel spectrograms
        
        Input: mel_spectrogram (batch, time, 80) - mel spectrogram from Tacotron
        Output: hidden_states (batch, seq_len, 768) - transformer features
        """
        # Move input to CPU (Mockingjay is always on CPU, enforced by to() override)
        # PyTorch requires inputs and model to be on the same device
        mel_cpu = mel_spectrogram.cpu()
        
        # Extract features using transformer encoder (bypasses waveform extracter)
        transformer = self.mockingjay.transformer
        last_hidden_state, _ = transformer._forward(mel_cpu)
        
        # Move features to GPU/MPS for classifier
        model_device = next(self.linear.parameters()).device
        hidden_states = last_hidden_state.to(model_device)
        
        # STEP 2: Pool sequence → single vector per sample
        pooled = self._pool_sequence(hidden_states, mask)
        
        # STEP 3: Classify
        return self.linear(pooled)
    
    def _forward_from_wav(self, wav: Tensor, mask: Tensor) -> Tensor:
        """
        STEP 1: Extract features from waveforms
        
        Input: wav (batch, max_time) - padded waveforms, mask (batch, max_time)
        Output: hidden_states (batch, seq_len, 768) - transformer features
        """
        # Move inputs to CPU (Mockingjay is always on CPU, enforced by to() override)
        # PyTorch requires inputs and model to be on the same device
        wav_cpu = wav.cpu()
        mask_cpu = mask.cpu()
        
        # Extract unpadded waveforms (Mockingjay expects unpadded list)
        # Use mask to determine actual length of each waveform
        wav_list = []
        for i in range(wav.shape[0]):
            actual_length = mask_cpu[i].sum().item()  # Count True values
            unpadded_wav = wav_cpu[i, :actual_length]  # Remove padding
            wav_list.append(unpadded_wav)
        
        # Extract features using Mockingjay
        outputs = self.mockingjay(wav_list)
        
        # Get hidden states (Mockingjay always returns dict with 'last_hidden_state')
        if not isinstance(outputs, dict) or 'last_hidden_state' not in outputs:
            raise ValueError(f"Unexpected Mockingjay output format: {type(outputs)}")
        
        # Move features to GPU/MPS for classifier
        model_device = next(self.linear.parameters()).device
        hidden_states = outputs['last_hidden_state'].to(model_device)
        
        # STEP 2: Pool sequence → single vector per sample
        pooled = self._pool_sequence(hidden_states, mask)
        
        # STEP 3: Classify
        return self.linear(pooled)
    
    def _pool_sequence(self, hidden_states: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        STEP 2: Pool sequence of features → single vector per sample
        
        Input: hidden_states (batch, seq_len, 768) - features for each time step
        Output: pooled (batch, 768) - single vector per sample
        
        Uses masked mean pooling: average over time, ignoring padding.
        """
        if mask is None:
            # Simple mean pooling if no mask
            return hidden_states.mean(dim=1)
        
        # Masked mean pooling
        # Expand mask to match hidden_states shape: (batch, seq_len) → (batch, seq_len, 1)
        # Move mask to same device as hidden_states
        mask_expanded = mask.unsqueeze(-1).float().to(hidden_states.device)
        
        # Adjust mask length to match hidden_states (they may differ due to downsampling)
        if mask_expanded.shape[1] != hidden_states.shape[1]:
            if mask_expanded.shape[1] < hidden_states.shape[1]:
                # Pad mask if shorter
                pad_len = hidden_states.shape[1] - mask_expanded.shape[1]
                padding = torch.zeros(mask_expanded.shape[0], pad_len, 1, device=mask_expanded.device)
                mask_expanded = torch.cat([mask_expanded, padding], dim=1)
            else:
                # Truncate mask if longer
                mask_expanded = mask_expanded[:, :hidden_states.shape[1], :]
        
        # Apply mask and compute mean
        masked_hidden = hidden_states * mask_expanded  # Zero out padding
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
        pooled = masked_hidden.sum(dim=1) / mask_sum  # Mean over time
        
        return pooled

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        # Set Mockingjay to train mode
        # If using LoRA, only LoRA parameters will be updated
        # If not using LoRA, only unfrozen layers (10 and 11) will be updated
        self.mockingjay.train()

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        self.log(
            "training_loss",
            loss.detach(),
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        # Update metrics
        self.f1_val(y_pred, trustworthy)
        self.acc_val(y_pred, trustworthy)

        # Log validation metrics
        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "validation_f1",
            self.f1_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "validation_accuracy",
            self.acc_val,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def on_train_start(self):
        """Called at the beginning of training before the first epoch."""
        super().on_train_start()
        # Ensure Mockingjay stays on CPU
        if hasattr(self, 'mockingjay'):
            self.mockingjay = self.mockingjay.cpu()
        max_epochs = self.trainer.max_epochs
        print(f"\n{'='*60}")
        print(f"Starting training with max_epochs = {max_epochs}")
        print(f"{'='*60}\n")

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        y_pred = self(wav=wav, mask=mask)
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=trustworthy)

        self.f1_test(y_pred, trustworthy)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test_f1",
            self.f1_test,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

