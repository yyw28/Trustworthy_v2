from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from tspeech.model.tts import TTSModel
from tspeech.data.tts import TTSDatamodule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(TTSModel, TTSDatamodule)


if __name__ == "__main__":
    cli_main()
