import torch
import lightning as L

from torchmetrics.collections import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


def choose(n: int, k: int) -> int:
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class Perceptron(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, bias: bool):
        super().__init__()

        self._in_size = in_size
        self._out_size = out_size
        self._bias = bias

        self._linear = torch.nn.Linear(self._in_size, self._out_size, bias=self._bias)
        self._batchnorm = torch.nn.BatchNorm1d(self._out_size)
        self._activation = torch.nn.ReLU()
        # self._dropout = torch.nn.Dropout1d(0.05)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        return self._activation(self._batchnorm(self._linear(input)))# self._dropout()


class MLP(torch.nn.Module):
    def __init__(self, in_size: int, layer_sizes: list[int], bias: bool = True):
        super().__init__()
        self._mlp = torch.nn.Sequential(*[
            Perceptron(
                layer_sizes[i - 1] if i > 0 else in_size,
                layer_sizes[i],
                bias,
            )
            for i in range(len(layer_sizes))
        ])

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        return self._mlp(input)


class DenseArch(torch.nn.Module):
    def __init__(self, in_features: int, layer_sizes: list[int]):
        super().__init__()
        self.model = MLP(in_features, layer_sizes, bias=True)

    def forward(self, dense_features: torch.FloatTensor) -> torch.FloatTensor:
        # B X D
        return self.model(dense_features)


class SparseArch(torch.nn.Module):
    def __init__(self, sparse_config):
        super().__init__()
        self.embedding_collection = torch.nn.ModuleDict({
            k : torch.nn.Embedding(
                num_embeddings=sparse_config[k]["num_embeddings"] + 1,
                embedding_dim=sparse_config[k]["embedding_dim"],
                padding_idx=0,
            )
            for k in sparse_config
        })
        self.sparse_feature_names = list(sparse_config.keys())

    def forward(self, sparse_features) -> torch.FloatTensor:
        embeddings = []
        for k in sparse_features:
            embeddings.append(embedding_collection[k](sparse_features[k]))
        return torch.hstack(embeddings)


class InteractionArch(torch.nn.Module):
    def __init__(self, num_sparse_features: int):
        super().__init__()
        self.F = num_sparse_features
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.F + 1, self.F + 1, offset=1),
            persistent=False,
        )

    def forward(self, dense_features: torch.FloatTensor, sparse_features: torch.LongTensor) -> torch.FloatTensor:
        if self.F <= 0:
            return dense_features

        (B, D) = dense_features.shape

        combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)

        interactions = torch.bmm(combined_values, torch.transpose(combined_values, 1, 2))
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]
        return torch.cat((dense_features, interactions_flat), dim=1)


class LowRankCrossNet(torch.nn.Module):
    def __init__(self, in_features: int, num_layers: int, low_rank: int):
        super().__init__()
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._low_rank = low_rank
        self.W_kernels = torch.nn.ParameterList([
            torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_features, self._low_rank)))
            for i in range(self._num_layers)
        ])
        self.V_kernels = torch.nn.ParameterList([
            torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self._low_rank, in_features)))
            for i in range(self._num_layers)
        ])
        self.bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features))) for i in range(self._num_layers)]
        )

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        x_0 = input
        x_l = x_0
        for layer in range(self._num_layers):
            x_l_v = torch.nn.functional.linear(x_l, self.V_kernels[layer])
            x_l_w = torch.nn.functional.linear(x_l_v, self.W_kernels[layer])
            x_l = x_0 * (x_l_w + self.bias[layer]) + x_l  # (B, N)
        return x_l


class InteractionDCNArch(torch.nn.Module):
    def __init__(self, num_sparse_features: torch.FloatTensor, crossnet: torch.nn.Module):
        super().__init__()
        self.F = num_sparse_features
        self.crossnet = crossnet

    def forward(self, dense_features: torch.FloatTensor, sparse_features: torch.LongTensor) -> torch.FloatTensor:
        if self.F <= 0:
            return dense_features

        (B, D) = dense_features.shape

        combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)
        return self.crossnet(combined_values.reshape([B, -1]))


class OverArch(torch.nn.Module):
    def __init__(self, in_features: int, layer_sizes: list[int]):
        super().__init__()

        self.model = torch.nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
            ),
            torch.nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True),
        )

    def forward(self, features: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(features)


class DLRM(torch.nn.Module):
    def __init__(
        self, embedding_dim, sparse_config, dense_config):
        super().__init__()
        if dense_arch_layer_sizes[-1] != embedding_dim:
            dense_arch_layer_sizes.append(embedding_dim)
        if over_arch_layer_sizes[-1] != 1:
            over_arch_layer_sizes.append(1)

        self.sparse_arch = SparseArch(sparse_config)
        self.dense_arch = DenseArch(dense_config)

        num_sparse_features = len(self.sparse_arch.sparse_feature_names)
        if dcn_num_layers is not None:
            crossnet = LowRankCrossNet(
                in_features=(num_sparse_features + 1) * embedding_dim,
                num_layers=dcn_num_layers,
                low_rank=dcn_low_rank_dim,
            )
            self.inter_arch = InteractionDCNArch(num_sparse_features=num_sparse_features, crossnet=crossnet)
            over_in_features = (num_sparse_features + 1) * embedding_dim
        else:

            self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
            over_in_features = embedding_dim + choose(num_sparse_features, 2) + num_sparse_features

        self.over_arch = OverArch(over_in_features, over_arch_layer_sizes)

    def forward(
        self, dense_features: torch.FloatTensor, sparse_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        embedded_dense = self.dense_arch(dense_features=dense_features)
        embedded_sparse = self.sparse_arch(sparse_features=sparse_features)
        concatenated_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
        logits = self.over_arch(concatenated_dense)
        return logits


class DLRMTrain(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.dlrm = DLRM(**config.dlrm)

        self.classification_metrics = MetricCollection({
            "precision": Precision(task="binary", average="weighted"),
            "recall": Recall(task="binary", average="weighted"),
            "f1": F1Score(task="binary", average="weighted"),
        })

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.hp.lr)
        return [optimizer]

    def forward(
        self,
        dense_features: torch.FloatTensor,
        sparse_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        prob = self.dlrm(dense_features=dense_features, sparse_features=sparse_features).sigmoid()
        return prob

    def training_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> torch.FloatTensor:
        loss = self._step(batch, "train")
        return loss

    def validation_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
    ) -> torch.FloatTensor:
        loss = self._step(batch, "valid")
        return loss


    def _step(
        self,
        batch: list[torch.Tensor],
        phase: str,
    ) -> torch.FloatTensor:
        dense_features, sparse_features, label, weight = batch
        prob = self(dense_features=dense_features, sparse_features=sparse_features)

        loss = torch.nn.functional.binary_cross_entropy(prob, label, weight)

        metric = self.classification_metrics(prob, label)
        metric = {f"{phase}_{k}": v for k, v in metric.items()}

        if phase == "train":
            on_step = True
            on_epoch = False

        else:
            on_step = False
            on_epoch = True

        self.log(f"{phase}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=True)
        self.log_dict(metric, on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=True)

        return loss
