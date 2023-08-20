import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics as M
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(pl.LightningModule):
    def __init__(self, hidden_dim: int, dropout: float = 0.5, lr: float = 0.001, reduce_lr_patience: int = 50):
        super().__init__()
        self.lr = lr
        self.reduce_lr_parience = reduce_lr_patience
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # activation function
            nn.LazyLinear(1),
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

    def shared_step(self, batch, batch_idx, name: str = "train"):
        x, y = batch
        y_pred = self.forward(x).float()
        y = y.float()
        loss = F.mse_loss(y_pred, y)
        r2 = M.functional.r2_score(y_pred, y)
        pearson = M.functional.pearson_corrcoef(y_pred, y)
        expvar = M.functional.explained_variance(y_pred, y)
        concord = M.functional.concordance_corrcoef(y_pred, y)
        self.log(f"{name}/loss", loss)
        self.log(f"{name}/r2", r2)
        self.log(f"{name}/pearson", pearson)
        self.log(f"{name}/expvar", expvar)
        self.log(f"{name}/concord", concord)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimisers = [optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [
            {
                "scheduler": ReduceLROnPlateau(
                    optimisers[0],
                    factor=0.1,
                    patience=self.reduce_lr_parience,
                    min_lr=1e-7,
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimisers, schedulers
