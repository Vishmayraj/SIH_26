"""Training entry point for Channel A - MIP Section 4.2.

Usage (from repo root, inside the `ml` container):
    python models/channel_a_velocity/train.py --config models/channel_a_velocity/config.yaml
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.channel_a_velocity.dataset import ChannelAVelocityDataset
from models.channel_a_velocity.model import ChannelAVelocityNet


class ChannelAModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        mcfg = cfg["model"]
        self.model = ChannelAVelocityNet(
            in_channels=mcfg["input_channels"],
            channels=mcfg["tcn_channels"],
            dilations=mcfg["tcn_dilations"],
            kernel_size=mcfg["kernel_size"],
            dropout=mcfg["dropout"],
        )
        self.loss_fn = nn.HuberLoss(delta=cfg["loss"]["delta"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(torch.mean((pred - y) ** 2))  # the real Section 4.2 target metric
        self.log("val_loss", loss)
        self.log("val_rmse_mps", rmse)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["optimizer"]["lr"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.hparams["optimizer"]["schedule_factor"],
            patience=self.hparams["optimizer"]["schedule_patience"],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    stats_path = "data/processed/channel_a_velocity/norm_stats.json"
    train_ds = ChannelAVelocityDataset("data/processed/channel_a_velocity", "train", stats_path)
    val_ds = ChannelAVelocityDataset("data/processed/channel_a_velocity", "val", stats_path)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    module = ChannelAModule(cfg)
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        gradient_clip_val=cfg["train"]["gradient_clip_max_norm"],
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["train"]["early_stopping_patience"],
            ),
            checkpoint,
        ],
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"best_checkpoint={checkpoint.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/channel_a_velocity/config.yaml")
    args = parser.parse_args()
    main(args.config)
