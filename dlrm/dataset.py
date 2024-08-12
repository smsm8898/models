import torch
import datetime
import lightning as L
import pandas as pd
import numpy as np


def process_features():
    pass


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx) -> list[torch.Tensor]:
        pass

    def _preprocess(self):
        process_features()


class DatasetModule(L.LightningDataModule):
    def __init__(self,):
        super().__init__()
        self.save_hyperparameters()
        pass

    def setup(self, stage=None):
        self.train_dataset = DataframeDataset()
        self.valid_dataset = DataframeDataset()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
