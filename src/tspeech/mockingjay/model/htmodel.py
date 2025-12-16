import torch
import torchmetrics
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F
from s3prl import hub


class HTModel(pl.LightningModule):
    def __init__(self, mockingjay_config_path: str, trainable_layers: int):
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
        
        # Freeze all Mockingjay parameters except the last two transformer layers
        # Mockingjay has 12 transformer layers (layer.0 through layer.11)
        # We'll unfreeze layers 10 and 11 (the last two layers)
        for name, param in self.mockingjay.named_parameters():
            # Unfreeze the last two transformer layers (layer.10 and layer.11)
            if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Get the output dimension from Mockingjay
        # Mockingjay typically outputs features of dimension 768
        # We'll determine this dynamically from the model
        with torch.no_grad():
            dummy_input = [torch.randn(16000)]  # 1 second of audio at 16kHz
            dummy_output = self.mockingjay(dummy_input)
            if isinstance(dummy_output, dict):
                # Get the hidden states dimension
                hidden_states = dummy_output.get('hidden_states', dummy_output.get('last_hidden_state'))
                if hidden_states is None:
                    hidden_states = list(dummy_output.values())[0]
                # Handle tuple, list, or tensor
                if isinstance(hidden_states, (tuple, list)):
                    hidden_states = hidden_states[0] if len(hidden_states) > 0 else hidden_states
                if hasattr(hidden_states, 'shape'):
                    hidden_size = hidden_states.shape[-1]
                else:
                    hidden_size = 768  # Default fallback
            else:
                if hasattr(dummy_output, 'shape') and dummy_output.dim() > 1:
                    hidden_size = dummy_output.shape[-1]
                else:
                    hidden_size = 768  # Default fallback
        
        self.linear = nn.Sequential(nn.Linear(hidden_size, 1))

        self.f1_val = torchmetrics.F1Score(task="binary")
        self.acc_val = torchmetrics.Accuracy(task="binary")
        self.f1_test = torchmetrics.F1Score(task="binary")

        self.trainable_layers = trainable_layers

    def forward(self, wav: Tensor, mask: Tensor) -> Tensor:
        """
        The model's forward pass

        Parameters
        ----------
        wav : Tensor
            A Tensor of shape (batch_size, sequence_length). Contains floating-point mono 16 kHz audio waveform data
        mask : Tensor
            A Tensor of shape (batch_size, sequence_length). Contains Boolean values indicating whether the corresponding element in wav is not masked (True) or masked (False)

        Returns
        -------
        Tensor
            A Tensor of shape (batch_size, 1) for binary classification.
        """
        # Mockingjay expects a list of waveforms (one per sample in batch)
        # Convert batch tensor to list more efficiently
        batch_size = wav.shape[0]
        wav_list = [wav[i].cpu() if wav.device.type != 'cpu' else wav[i] for i in range(batch_size)]
        
        # Get features from Mockingjay
        # Mockingjay returns a dictionary with 'hidden_states' key
        with torch.cuda.amp.autocast(enabled=False):  # Mockingjay handles its own precision
            outputs = self.mockingjay(wav_list)
        
        # Extract hidden states (typically the last layer)
        # Mockingjay returns dict with 'hidden_states' (list) and 'last_hidden_state' (tensor)
        if isinstance(outputs, dict):
            # Use 'last_hidden_state' if available (already stacked), otherwise use 'hidden_states'
            if 'last_hidden_state' in outputs:
                hidden_states = outputs['last_hidden_state']
            elif 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states']
                # If it's a list of tensors (one per sample), we need to pad and stack
                if isinstance(hidden_states, list):
                    # Get max length
                    max_len = max(h.shape[0] for h in hidden_states)
                    hidden_dim = hidden_states[0].shape[-1]
                    # Pad and stack
                    padded_hidden = []
                    for h in hidden_states:
                        pad_len = max_len - h.shape[0]
                        if pad_len > 0:
                            h = torch.cat([h, torch.zeros(pad_len, hidden_dim, device=h.device)], dim=0)
                        padded_hidden.append(h)
                    hidden_states = torch.stack(padded_hidden)
            else:
                # If no expected keys, try to get the first value
                hidden_states = list(outputs.values())[0]
        else:
            hidden_states = outputs
        
        # Ensure hidden_states is a tensor, not a list
        if isinstance(hidden_states, list):
            # If still a list, pad and stack
            max_len = max(h.shape[0] for h in hidden_states)
            hidden_dim = hidden_states[0].shape[-1]
            padded_hidden = []
            for h in hidden_states:
                pad_len = max_len - h.shape[0]
                if pad_len > 0:
                    h = torch.cat([h, torch.zeros(pad_len, hidden_dim, device=h.device)], dim=0)
                padded_hidden.append(h)
            hidden_states = torch.stack(padded_hidden)
        
        # Pool the hidden states (mean pooling)
        # Apply mask if needed
        if hidden_states.dim() == 3:  # (batch, seq_len, hidden_dim)
            # Apply mask for mean pooling
            # Create mask based on actual sequence lengths
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            # Adjust mask to match hidden_states sequence length
            if mask_expanded.shape[1] != hidden_states.shape[1]:
                # Interpolate or pad mask to match
                if mask_expanded.shape[1] < hidden_states.shape[1]:
                    # Pad mask
                    pad_len = hidden_states.shape[1] - mask_expanded.shape[1]
                    mask_expanded = torch.cat([mask_expanded, torch.zeros(mask_expanded.shape[0], pad_len, 1, device=mask_expanded.device)], dim=1)
                else:
                    # Truncate mask
                    mask_expanded = mask_expanded[:, :hidden_states.shape[1], :]
            
            masked_hidden = hidden_states * mask_expanded
            pooled_output = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled_output = hidden_states.mean(dim=1) if hidden_states.dim() > 1 else hidden_states
        
        return self.linear(pooled_output)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        # Set Mockingjay to train mode to allow training of unfrozen layers (10 and 11)
        # Frozen layers (0-9) won't update due to requires_grad=False
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

