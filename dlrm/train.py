import torch
import pickle
import argparse
import datetime
import dataclasses
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, RichProgressBar


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--accelerator", type=str)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    return parser.parse_args()


@dataclasses.dataclass
class RichProgressBarTheme:
    description = "white"
    progress_bar = "#6206E0"
    progress_bar_finished = "#6206E0"
    progress_bar_pulse = "#6206E0"
    batch_progress = "white"
    time = "black"
    processing_speed = "black"
    metrics = "red"
    metrics_text_delimiter = "\n"
    metrics_format = ".4f"


if __name__ == "__main__":
    args = get_arguments()
    config = get_config(args)
    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    date = datetime.date.fromisoformat(args.date)

    with open("", "rb") as f:
        vocab = pickle.load(f)

    callbacks = [
        ModelSummary(max_depth=1),
        EarlyStopping(monitor="valid_precision", mode="max", patience=3, verbose=False),
        ModelCheckpoint(
            dirpath="",
            monitor="valid_precision",
            mode="max",
        ),
        RichProgressBar(theme=RichProgressBarTheme),
    ]

    dm = DataframeDatasetModule()

    dlrm = DLRMTrain(config)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        callbacks=callbacks,
        enable_model_summary=False,
    )

    trainer.fit(dlrm, dm)
