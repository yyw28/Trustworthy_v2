from os import path

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


class TISDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: str, sr: int = 16000):
        self.df = df
        self.dataset_dir = dataset_dir

        self.resample = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sr)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        data = self.df.iloc[i].to_dict()

        wav, _ = torchaudio.load(
            path.join(
                self.dataset_dir,
                "Speech WAV Files",
                f"{data['Speaker_Ethnicity'].replace('_', ' ')} {data['Speaker_AgeGroup']}",
                f"{data['Audio_Filename'].strip()}.wav",
            )
        )
        wav = self.resample(wav)

        mask = torch.ones_like(wav, dtype=torch.bool)
        trustworthy = torch.tensor(
            [[data["Speaker_Intent"] == "Trustworthy"]], dtype=torch.float
        )

        return wav, mask, trustworthy
