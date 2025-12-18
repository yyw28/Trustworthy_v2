import os
from os import path
from typing import Final, Literal, Union, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from tspeech.data.collate_fn import trustworthy_collate_fn
from tspeech.data.tis_dataset import TISDataset
from tspeech.data.cxd_dataset import CXDDataset
from tspeech.data.combined_dataset import CombinedDataset


class HTDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: dict[Union[Literal["tis"], Literal["synth"], Literal["cxd"], Literal["combined"]], str],
        combined_csv: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
    ):

        super().__init__()

        self.num_workers = num_workers
        self.batch_size: Final[int] = batch_size
        self.combined_csv = combined_csv

        if len(datasets) == 0 and combined_csv is None:
            raise Exception("At least one dataset must be specified or combined_csv must be provided")

        self.datasets: list[Dataset] = []

        # If combined_csv is provided, use it instead of individual datasets
        if combined_csv is not None:
            if not os.path.exists(combined_csv):
                raise FileNotFoundError(f"Combined CSV file not found: {combined_csv}")
            
            # Use the datasets dict as base directories mapping
            base_dirs = datasets.copy()
            self.datasets.append(CombinedDataset(csv_path=combined_csv, base_dirs=base_dirs))
        else:
            # Load individual datasets (original behavior)
            if "tis" in datasets:
                dataset_dir = datasets["tis"]

                # Some of the wav files are missing. Search the ones that are there and use it to narrow down the dataframe
                wav_files: set[str] = set()
                for _, _, files in os.walk(path.join(dataset_dir, "Speech WAV Files")):
                    for file in files:
                        if file.endswith(".wav"):
                            wav_files.add(file.split(".")[0])

                df = pd.read_csv(
                    path.join(dataset_dir, "Speech_dataset_characteristics.csv")
                )
                df["Audio_Filename"] = df["Audio_Filename"].str.strip()
                df = df[df["Audio_Filename"].isin(wav_files)]

                self.datasets.append(TISDataset(df=df, dataset_dir=dataset_dir))

            if "cxd" in datasets:
                dataset_dir = datasets["cxd"]

                # Load CXD CSV file
                csv_path = path.join(dataset_dir, "question_responses_with_trust_ratings.csv")
                df = pd.read_csv(csv_path)
                
                # Verify that required columns exist
                if "name" not in df.columns or "trust_label" not in df.columns:
                    raise ValueError(
                        "CXD dataset CSV must contain 'name' and 'trust_label' columns"
                    )
                
                # Filter out rows where trust_label = -1
                df = df[df['trust_label'] != -1]
                
                # Filter to only rows where we can find corresponding wav files
                wav_dir = path.join(dataset_dir, "wav")
                if os.path.exists(wav_dir):
                    wav_files: set[str] = set()
                    for file in os.listdir(wav_dir):
                        if file.endswith(".wav"):
                            wav_files.add(file)
                    
                    # Filter dataframe to only include rows where wav file exists
                    # Try both direct match and base part match
                    def has_wav_file(name: str) -> bool:
                        wav_filename = name.replace('.txt', '.wav')
                        if wav_filename in wav_files:
                            return True
                        # Try base part matching
                        base_part = name.split(':')[0] if ':' in name else name
                        return any(base_part in f for f in wav_files)
                    
                    df = df[df['name'].apply(has_wav_file)]
                
                self.datasets.append(CXDDataset(df=df, dataset_dir=dataset_dir))

    def setup(self, stage: str):
        datasets_train: list[Dataset] = []
        datasets_val: list[Dataset] = []
        datasets_test: list[Dataset] = []

        for dataset in self.datasets:
            # First split: 80% train+val, 20% test
            dataset_train_val, dataset_test = random_split(
                dataset,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
            
            # Second split: 80% train, 20% validation (from the 80% train+val)
            dataset_train, dataset_val = random_split(
                dataset_train_val,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(43),
            )

            datasets_train.append(dataset_train)
            datasets_val.append(dataset_val)
            datasets_test.append(dataset_test)

        self.dataset_train = ConcatDataset(datasets_train)
        self.dataset_val = ConcatDataset(datasets_val)
        self.dataset_test = ConcatDataset(datasets_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            collate_fn=trustworthy_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

