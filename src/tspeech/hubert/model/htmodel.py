import torchmetrics
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from transformers import HubertModel


class HTModel(pl.LightningModule):
    def __init__(self, hubert_model_name: str, trainable_layers: int):
        super().__init__()
        self.save_hyperparameters()

        self.hubert = HubertModel.from_pretrained(hubert_model_name)
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

        for t in range(1, self.trainable_layers + 1):
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
