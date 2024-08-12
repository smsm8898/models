import torch
import pandas as pd
import numpy as np
import torchrec
import lightning as L
from torchmetrics import AUROC
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)
from torchmetrics.collections import MetricCollection
from typing import List, Dict, Union, Optional




class TwHINDataset(torch.utils.data.Dataset):
    lhs_column_name = "lhs"
    rhs_column_name = "rhs"
    rel_column_name = "rel"
    def __init__(self, df, shuffle=False):
        if shuffle:
            df = df.sample(frac=1)
        self.df = df
        self.lhs = torch.from_numpy(df[self.lhs_column_name].to_numpy(dtype=np.int64))
        self.rhs = torch.from_numpy(df[self.rhs_column_name].to_numpy(dtype=np.int64))
        self.rel = torch.from_numpy(df[self.rel_column_name].to_numpy(dtype=np.int64))
        self.label = torch.ones_like(self.lhs, dtype=torch.float32)
        self.weight = torch.ones_like(self.lhs, dtype=torch.float32)
        
    def __len__(self):
        return len(self.lhs)
    
    def __getitem__(self, idx):
        inputs = {
            "lhs":self.lhs[idx],
            "rhs":self.rhs[idx],
            "rel":self.rel[idx],
            "label":self.label[idx],
            "weight":self.weight[idx],
        }
        return inputs
    
class TwHINDataModule(L.LightningDataModule):
    def __init__(
        self,
        data:Union[pd.DataFrame, str],
        batch_size:Optional[int]=None,
        num_workers:Optional[int]=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, str):
            self.df = pd.read_parquet(data)
        else:
            raise ValueError("data must be either pd.DataFrame or str")
            
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = TwHINDataset(self.df, shuffle=True)
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05])
        else:
            raise ValueError(f"stage {stage} is not supported")
            
        
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )