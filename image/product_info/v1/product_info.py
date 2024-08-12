import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import transformers
import torchmetrics
from PIL import Image
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from transformers import AutoConfig, AutoModel
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict


class ImageModelForMultiOuputClassificationPipeline:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.vocab = self.model.vocab
        self.preprocessor = self.vocab["preprocessor"]
        self.target_list = ["category", "color"]

    def preprocess(self, hint, image):
        hint = torch.LongTensor([[self.vocab["hint"]["label2id"]]])
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        model_input = self.preprocessor(image)
        if model_input.ndim == 3:
            model_input = model_input.unsqueeze(dim=0)
        return {"hint": hint, "pixel_values": model_input}

    def predict(self, model_inputs):
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=3):
        result = defaultdict()
        for key, logits in model_outputs.items():
            if key == "loss":
                continue
            target_name, is_logits = key.split("_")
            if is_logits == "logits":
                probability = logits.softmax(-1)[0]
                scores, ids = probability.topk(top_k)
                result[f"{target_name}_scores"] = scores.tolist()
                result[f"{target_name}_ids"] = [self.vocab[target_name]["id2label"][id] for id in ids.tolist()]
        return result

    def inference(self, hint, image):
        try:
            ### inference ###
            model_inputs = self.preprocess(hint, image)
            model_outputs = self.predict(model_inputs)
            outputs = self.postprocess(model_outputs)

            result = defaultdict(dict)
            for target in self.target_list:
                for target_id, target_score in zip(outputs[f"{target}_ids"], outputs[f"{target}_scores"]):
                    result[target][target_id] = target_score

            return [result[k] for k in result]
        except:
            return None


def get_metrics(num_classes: int, label: str, mode: str):
    return MetricCollection({
        f"{mode}_acc_{label}": Accuracy(task="multiclass", num_classes=num_classes),
        f"{mode}_f1_{label}": F1Score(task="multiclass", num_classes=num_classes, average="weighted"),
    })


class ImageModelForMultiOuputClassification(pl.LightningModule):
    def __init__(
        self,
        vocab: dict,
        checkpoint: str,
        lr=5e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.vocab = vocab

        # num labels
        self.num_categories = ...
        self.num_color = ...
        self.num_hint = ...

        # pretrained image embedding model
        self.image_embedding_config = AutoConfig.from_pretrained(checkpoint)
        self.image_embedding_dim = self.image_embedding_config.hidden_size
        self.image_embedding_model = AutoModel.from_pretrained(checkpoint)

        # classifier head
        self.category_classifier = torch.nn.Linear(
            self.image_embedding_dim, self.num_category_labels
        )
        self.color_classifier = torch.nn.Linear(
            self.image_embedding_dim, self.num_color_labels
        )

        # metrics
        self.train_metrics_category = get_metrics(num_classes=self.num_category_labels, label="category", mode="train")
        self.train_metrics_color = get_metrics(num_classes=self.num_color_labels, label="color", mode="train")

        self.valid_metrics_category = get_metrics(num_classes=self.num_category_labels, label="category", mode="valid")
        self.valid_metrics_color = get_metrics(num_classes=self.num_color_labels, label="color", mode="valid")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]

        # metrics
        category_metric = self.train_metrics_category(output["category_logits"], batch["category"])
        color_metric = self.train_metrics_color(output["color_logits"], batch["color"])

        # logging
        self.log_dict(category_metric, on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict(color_metric, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]

        # metrics
        category_metric = self.valid_metrics_category(output["category_logits"], batch["category"])
        color_metric = self.valid_metrics_color(output["color_logits"], batch["color"])

        # logging
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(category_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(color_metric, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(**batch)

    def forward(
        self,
        hint: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        category: Optional[torch.Tensor] = None,
        color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.FloatTensor]:
        # Image Embedding
        outputs = self.image_embedding_model(pixel_values)
        sequence_output = outputs.last_hidden_state[:, 0, :]

        # One-Hot
        hint_oh = torch.nn.functional.one_hot(hint, num_classes=self.hint).view(-1, self.num_hint)

        # Concatenate
        sequence_output = torch.concat([sequence_output, hint_oh], axis=1)

        # Classifier Head(Logits)
        category_logits = self.category_classifier(sequence_output)
        color_logits = self.color_classifier(sequence_output)

        # LOSS
        category_loss = None
        color_loss = None
        loss = None
        if (category is not None) and (color is not None):
            category_loss = F.cross_entropy(
                category_logits.view(-1, self.num_category_labels),
                category.view(-1),
            )
            color_loss = F.cross_entropy(color_logits.view(-1, self.num_color_labels), color.view(-1))
            loss = 0.5 * category_loss + 0.5 * color_loss

        return {
            # logits
            "category_logits": category_logits,
            "color_logits": color_logits,
            # loss
            "loss": loss,
            "category_loss": category_loss,
            "color_loss": color_loss,
        }


class GRIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df[["image_path", "category", "color", "hint"]]
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[0]
        input_dict = row[1:].astype("int").to_dict()
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        input_dict["pixel_values"] = image
        return input_dict
