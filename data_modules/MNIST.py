from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class MNISTDataModule(pl.LightningDataModule):
    """lightning MNIST datamodule with following transforms
    (Resize:28x28->32x32, ToTensor, Normalize into [-1,1])
    """

    def __init__(self, data_dir: str = "./downloads"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = Compose([Resize(32), ToTensor(), Normalize(0.5, 0.5)])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage in ("fit", "validate", None):
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # use 20% of training data for validation
            total_size = len(mnist_full)
            train_size = int(total_size * 0.9)
            valid_size = total_size - train_size
            seed = torch.Generator().manual_seed(42)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [train_size, valid_size], generator=seed
            )

        # Assign test dataset for use in dataloader(s)
        if stage in ("test", None):
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage in ("predict", None):
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=12)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32, num_workers=12)
