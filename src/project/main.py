from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from project.model import ExampleModel
from project.data import ExampleDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(ExampleModel, ExampleDataModule)


if __name__ == "__main__":
    cli_main()
