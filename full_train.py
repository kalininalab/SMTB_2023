import os
import sys

import wandb
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.model import Model
from src.data import FluorescenceDataset, train_validation_test

# technical setting to make sure, parallelization works if multiple models are trained in parallel
torch.multiprocessing.set_sharing_strategy("file_system")

HIDDEN_DIM = 512
BATCH_SIZE = 1024
MAX_EPOCH = 10000
DROPOUT = 0.2
DATASET = "/shared/stability"
EARLY_STOPPING_PATIENCE = 100
LR = 0.001
REDUCE_LR_PATIENCE = 50


def train(model_name: str, layer_num: int):
    # for reproducibility
    seed_everything(42)

    # define the logger
    logger = WandbLogger(
        log_model=True,
        project=DATASET.split("/")[-1],
        entity="smtb2023",
        name=f"{model_name.split('_')[1]}_{layer_num:02d}",
        config={
            "model_name": model_name,
            "layer_num": layer_num,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "lr": LR,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
        },
    )

    # define the callbacks with EarlyStopping and two more for nicer tracking
    callbacks = [
        EarlyStopping(monitor="val/loss", patience=EARLY_STOPPING_PATIENCE, mode="min"),
        RichModelSummary(),
        RichProgressBar(),
    ]

    # define the Trainer and it's most important arguments
    trainer = Trainer(
        devices=1,
        max_epochs=MAX_EPOCH,
        callbacks=callbacks,
        logger=logger,
    )

    # initialize the model
    model = Model(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

    # look into the directory below
    datasets = []
    for ds in ["train", "validation", "test"]:
        p = Path(DATASET) / model_name / ds
        datasets.append(train_validation_test(p, layer_num))

    train_dataset = FluorescenceDataset(datasets[0][0], datasets[0][1])
    validation_dataset = FluorescenceDataset(datasets[1][0], datasets[1][1])
    test_dataset = FluorescenceDataset(datasets[2][0], datasets[2][1])

    train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4096)
    test_dataloader = DataLoader(test_dataset, batch_size=4096)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

    # fit and test the (best) model
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)


model_names = {
    48: "esm2_t48_15B_UR50D",
    36: "esm2_t36_3B_UR50D",
    33: "esm2_t33_650M_UR50D",
    30: "esm2_t30_150M_UR50D",
    12: "esm2_t12_35M_UR50D",
    6: "esm2_t6_8M_UR50D",
}

num_layers = int(sys.argv[1])
model_name = model_names[num_layers]
for layer in range(num_layers + 1):
    # wandb.init()
    train(model_name, layer)
    wandb.finish()
