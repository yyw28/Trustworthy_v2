import torch
from torch import Tensor
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> Tensor:
        return torch.tensor([0.0, 0.0, 0.0])
