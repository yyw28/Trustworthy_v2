from lightning import pytorch as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ExampleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # Example model code - delete this
        self.linear = nn.Linear(3, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Example model code - delete this
        return self.linear(x)

    def training_step(self, batch, batch_idx: int):
        # Example model code - delete this
        out = self(batch)
        return F.mse_loss(out, torch.ones((batch.shape[0], 1), device=batch.device))

    def validation_step(self, batch, batch_idx: int):
        # Example model code - delete this
        out = self(batch)
        return F.mse_loss(out, torch.ones((batch.shape[0], 1), device=batch.device))

    def test_step(self, batch, batch_idx: int):
        # Example model code - delete this
        out = self(batch)
        return F.mse_loss(out, torch.ones((batch.shape[0], 1), device=batch.device))
