import torchmetrics
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from transformers import HubertModel
import os

# Force use of safetensors to avoid torch.load vulnerability issue
os.environ["TRANSFORMERS_SAFE_LOADING"] = "1"


class HTModel(pl.LightningModule):
    def __init__(self, hubert_model_name: str, trainable_layers: int):
        super().__init__()
        self.save_hyperparameters()

        # Use safetensors to avoid PyTorch 2.6+ requirement
        # Try safetensors first, fallback to regular if not available
        try:
            self.hubert = HubertModel.from_pretrained(
                hubert_model_name,
                use_safetensors=True,
            )
        except Exception as e:
            # If safetensors not available, try without (may require PyTorch 2.6+)
            print(f"Warning: Could not load safetensors, trying regular format: {e}")
            # For now, raise error - user needs to upgrade PyTorch or use model with safetensors
            raise ValueError(
                f"Model loading failed. Please either:\n"
                f"1. Upgrade PyTorch to 2.6+: pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121\n"
                f"2. Or ensure model has safetensors format available.\n"
                f"Original error: {e}"
            )
        self.linear = nn.Sequential(nn.Linear(self.hubert.config.hidden_size, 1))

        self.f1_val = torchmetrics.F1Score(task="binary")
        self.acc_val = torchmetrics.Accuracy(task="binary")
        self.f1_test = torchmetrics.F1Score(task="binary")

        self.trainable_layers = trainable_layers
        
        # Freeze all HuBERT parameters initially
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer layers
        num_layers = len(self.hubert.encoder.layers)
        for t in range(1, trainable_layers + 1):
            if t <= num_layers:
                for param in self.hubert.encoder.layers[-t].parameters():
                    param.requires_grad = True
        
        # Always train the classifier
        for param in self.linear.parameters():
            param.requires_grad = True
        
        # Set trainable layers to train mode (others stay in eval)
        self.hubert.eval()  # Set base model to eval
        for t in range(1, trainable_layers + 1):
            if t <= num_layers:
                self.hubert.encoder.layers[-t].train()  # Set trainable layers to train mode

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return Adam(self.parameters(), lr=0.000001)

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
        outputs = self.hubert(input_values=wav, attention_mask=mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.linear(pooled_output)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        wav, mask, trustworthy = batch
        batch_size = wav.shape[0]

        # Ensure trainable layers are in train mode
        num_layers = len(self.hubert.encoder.layers)
        for t in range(1, self.trainable_layers + 1):
            if t <= num_layers:
                self.hubert.encoder.layers[-t].train()

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

        self.f1_val(y_pred, trustworthy)
        self.acc_val(y_pred, trustworthy)

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
