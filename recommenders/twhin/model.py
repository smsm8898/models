import math
import torch
import torchrec
import lightning as L
import numpy as np
import pandas as pd

from torchmetrics.collections import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)
from torchmetrics import AUROC
from typing import List, Dict, Union


class IntegerLookup(torch.nn.Module):
    def __init__(
        self,
        vocab,
    ):
        super().__init__()
        self.vocab = vocab
        self.max = max(self.vocab)+1
        _weight = torch.zeros(self.max).long()
        _weight[self.vocab] = torch.arange( # index
            start=1,
            end=len(self.vocab)+1, 
            dtype=_weight.dtype,
        )
        self.lookup = torch.nn.Embedding(
            num_embeddings=self.max,
            embedding_dim=1,
            _weight=_weight.reshape(-1, 1),
            _freeze=True,
        )
        
    def forward(
        self,
        x: Union[torch.IntTensor, torch.LongTensor]
    ) -> Union[torch.IntTensor, torch.LongTensor]:
        x[x >= self.max] = 0
        return self.lookup(x)
    
    @property
    def device(self):
        return self.lookup.weight.device

class TwHIN(torch.nn.Module):
    def __init__(
        self,
        node_config:dict,
        embedding_dim:int,
        relations_type:List[int],
        table_names:List[List[str]],
        in_batch_negatives:int,
        device:Union[str, torch.device],
    ):
        super().__init__()
        self.node_config = node_config
        self.relations_t = torch.tensor(relations_type, device=device)
        self.num_relations = len(relations_type)
        self.num_tables = len(node_config)
        self.table_names = table_names
        self.embedding_dim = embedding_dim
        self.in_batch_negatives = in_batch_negatives
        self.device = device
        
        for name, unique in self.node_config.items():
            print(name, len(unique))

        self.tables = [
            torchrec.EmbeddingBagConfig(
                num_embeddings=len(unique)+1,
                embedding_dim=embedding_dim,
                name=name,
                feature_names=[name],
            ) for name, unique in self.node_config.items()
        ]
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.ebc = torchrec.EmbeddingBagCollection(tables=self.tables, device=self.device)
        self.all_trans_embs = torch.nn.parameter.Parameter(
            torch.nn.init.uniform_(
                torch.empty(self.num_relations, self.embedding_dim)
            )
        )
        self.lookup_tables = {}
        for i, name in enumerate(self.table_names):
            self.lookup_tables[i] = IntegerLookup(self.node_config[name], device=self.device)
        
    def _to_kjt(self, lhs, rhs, rel, batch_size):
        index = torch.LongTensor([1, 0, 2, 3])
        if self.relations_t.device != lhs.device:
            self.relations_t = self.relations_t.to(lhs.device)
        lookups = torch.concat((lhs[:, None], self.relations_t[rel], rhs[:, None]), dim=1)
        lookups = lookups[:, index].reshape(2*batch_size, 2)


        _, indices = torch.sort(lookups[:, 0], dim=0, stable=True)
        values = lookups[indices]                
        for name_index in self.lookup_tables:
            mask = (values[:, 0] == name_index)
            self.lookup_tables[name_index] = self.lookup_tables[name_index].to(values.device)
            values[mask, 1] = self.lookup_tables[name_index](values[mask, 1]).flatten()
        values = values[:, 1].int()


        lengths = torch.arange(self.num_tables, device=lookups.device)[:, None].eq(lookups[:, 0])
        lengths = lengths.flatten().int()

        return torchrec.KeyedJaggedTensor(keys=self.table_names, values=values, lengths=lengths)
        
        
    def forward(self, lhs, rhs, rel):
        # B: Batch_size
        # T: Num_tables(user, content, seller)
        # D: Embedding_dim
        
        
        batch_size = lhs.shape[0]
        nodes = self._to_kjt(lhs, rhs, rel, batch_size)
        
        outs = self.ebc(nodes)
        
        # 2B X TD
        x = outs.values()
        
        # 2B X T X D
        x = x.reshape(2*batch_size, -1, self.embedding_dim)
        
        # 2B X D
        x = torch.sum(x, 1) 
        
        # B X 2 X D
        x = x.reshape(batch_size, 2, self.embedding_dim) # 2는 lhs[0], rhs[1]을 의미함
        
        # translated
        trans_embs = self.all_trans_embs.data[rel]
        translated = x[:, 1, :] + trans_embs
        
        
        # in batch negative sample
        negs = []
        if self.in_batch_negatives:
            for relation in range(self.num_relations):
                rel_mask = (rel == relation)
                rel_count = rel_mask.sum()
                
                if not rel_count:
                    continue
                
                # R X D
                lhs_matrix = x[rel_mask, 0, :]
                rhs_matrix = x[rel_mask, 1, :]
                
                
                lhs_perm = torch.randperm(lhs_matrix.shape[0])
                # repeat until we have enough negatives
                lhs_perm = lhs_perm.repeat(math.ceil(float(self.in_batch_negatives) / rel_count))
                lhs_indices = lhs_perm[: self.in_batch_negatives]
                sampled_lhs = lhs_matrix[lhs_indices]
                
                
                rhs_perm = torch.randperm(rhs_matrix.shape[0])
                # repeat until we have enough negatives
                rhs_perm = rhs_perm.repeat(math.ceil(float(self.in_batch_negatives) / rel_count))
                rhs_indices = rhs_perm[: self.in_batch_negatives]
                sampled_rhs = rhs_matrix[rhs_indices]
                
                # RS - 기존 lhs(or rhs)와 sampled_rhs(in batch개 ...)랑 dot product
                # 즉, positive 1개당 in batch개의 negative sample을 학습
                # [B, D] * [in_batch_negatives, D]
                negs_rhs = torch.flatten(torch.matmul(lhs_matrix, sampled_rhs.t()))
                negs_lhs = torch.flatten(torch.matmul(rhs_matrix, sampled_lhs.t()))
                
                negs.append(negs_lhs)
                negs.append(negs_rhs)
        
        # Dot Product for positive
        x = (x[:, 0, :]*translated).sum(-1)
        
        # concat positives and negtives
        x = torch.cat([x, *negs])
        
        return x 



class TwHINModule(L.LightningModule):
    def __init__(self, twhin:TwHIN, lr):
        super().__init__()
        self.save_hyperparameters(ignore=["twhin"])
        self.twhin = twhin
        self.lr = lr

        self.metrics = MetricCollection({
            "precision":Precision(task="binary", average="weighted"),
            "recall":Recall(task="binary",  average="weighted"),
            "f1":F1Score(task="binary", average="weighted"),
        })
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr) 
        return [optimizer]
    
    def forward(self, batch):
        return self.twhin(**batch)
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")
    
    def predict_step(self, batch, batch_idx):
        in_batch_negatives = self.twhin.in_batch_negatives
        self.twhin.in_batch_negatives = 0
        probability = self.twhin(lhs=batch["lhs"], rhs=batch["rhs"], rel=batch["rel"]).sigmoid()
        self.twhin.in_batch_negatives = in_batch_negatives
        return probability
    
    def _step(self, batch, phase):
        label = batch.pop("label")
        weight = batch.pop("weight")
        batch_size = label.shape[0]
        
        logits = self(batch)
    
        num_negatives = 2 * batch_size * self.twhin.in_batch_negatives
        num_positives = batch_size
        
        neg_weight = num_positives / num_negatives + 0.5
        
        label = torch.cat([
            label.float(),
            torch.zeros(num_negatives, dtype=torch.float, device=logits.device)
        ])
        
        weight = torch.cat([
            weight.float(),
            torch.ones(num_negatives, dtype=torch.float, device=logits.device)*neg_weight
        ])
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label, weight)
        metric = self.metrics(logits, label)
        metric = {f"{phase}_{k}":v for k, v in metric.items()}
        if phase == "train":
            on_step=True
            on_epoch=False
        else:
            on_step=False
            on_epoch=True
        self.log(f"{phase}_loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=True)
        self.log_dict(metric, on_step=on_step, on_epoch=on_epoch, prog_bar=True, sync_dist=True)
        
        return loss
    