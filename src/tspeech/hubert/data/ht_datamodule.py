import os
from os import path
from typing import Final, Literal, Union, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from tspeech.data.collate_fn import trustworthy_collate_fn
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

        self.datasets: list[Dataset] = []

        # Use the datasets dict as base directories mapping
        base_dirs = datasets.copy()
        self.datasets.append(CombinedDataset(csv_path=combined_csv, base_dirs=base_dirs))

    def setup(self, stage: str):
        datasets_train: list[Dataset] = []
        datasets_val: list[Dataset] = []
        datasets_test: list[Dataset] = []

        for dataset in self.datasets:
            dataset_train, dataset_val, dataset_test = random_split(
                dataset,
                [0.7, 0.1, 0.2],
                generator=torch.Generator().manual_seed(42),
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

