import os
import pickle
import torch
import lightning as L
from lightning.pytorch import seed_everything

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, RichProgressBar

from model import TwHINModule, TwHIN
from datamodules import TwHINDataModule
from pprint import pprint

torch.set_float32_matmul_precision("high")
seed_everything(789)

callbacks = [
    ModelSummary(max_depth=1),
    EarlyStopping(
        monitor="valid_f1",
        mode="max",
        patience=2,
        verbose=False
    ),
    ModelCheckpoint(
        dirpath="",
        monitor="train_f1",
        mode="max",
        filename="TwHIN",
        enable_version_counter=False,
        save_last=True,
        auto_insert_metric_name=False,
    ),
    RichProgressBar()
]
logger = CSVLogger(save_dir="outputs/logs", name="twhin")

lr = 0.0005
data_config = {
    "data":"graph.parquet",
    "batch_size":1024,
    "num_workers":1,
}
with open("config.pkl", "rb") as f:
    backbone_model_config = pickle.load(f)
    
trainer_config = {
    "max_epochs":3,
    "accelerator":"cuda",
    "devices":[0],
    "enable_model_summary":"False",
    "callbacks":callbacks,
    "logger":logger,
}

if __name__ == "__main__":
    pprint(data_config)
    pprint(backbone_model_config)
    pprint(trainer_config)
    
    twhin_datamodule = TwHINDataModule(**data_config)
    
    twhin = TwHIN(**backbone_model_config)
    twhin_module = TwHINModule(twhin, lr)

    trainer = L.Trainer(**trainer_config)
    trainer.fit(twhin_module, twhin_datamodule)