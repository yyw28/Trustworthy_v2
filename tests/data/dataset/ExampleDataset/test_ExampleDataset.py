import pytest
import torch

from project.data.dataset import ExampleDataset


@pytest.fixture
def dataset():
    return ExampleDataset(dataset_dir=".")


def test_ExampleDataset_len(dataset):
    assert len(dataset) == 1


def test_ExampleDataset_getitem(dataset):
    torch.testing.assert_close(dataset[0], torch.tensor([0.0, 0.0, 0.0]))
