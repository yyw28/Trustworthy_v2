from typing import Final

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from project.data.dataset import ExampleDataset


class ExampleDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.batch_size: Final[int] = batch_size

    def prepare_data(self):
        print("Preparing data")

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.dataset_train = ExampleDataset(dataset_dir=self.dataset_dir)
                self.dataset_validate = ExampleDataset(dataset_dir=self.dataset_dir)
            case "validate":
                self.dataset_validate = ExampleDataset(dataset_dir=self.dataset_dir)
            case "test":
                self.dataset_test = ExampleDataset(dataset_dir=self.dataset_dir)
            case "predict":
                self.dataset_predict = ExampleDataset(dataset_dir=self.dataset_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_validate, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_predict, batch_size=self.batch_size)
