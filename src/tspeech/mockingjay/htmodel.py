from lightning.pytorch.cli import LightningCLI
import torch

# Mockingjay implementation with 80/20 train/test split
from tspeech.mockingjay.model.htmodel import HTModel
from tspeech.mockingjay.data.ht_datamodule import HTDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(HTModel, HTDataModule)


if __name__ == "__main__":
    cli_main()

