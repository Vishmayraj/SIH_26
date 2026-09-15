"""Training entry point for Channel B - MIP Section 4.3.

CarSpeedNet-style accelerometer-only velocity regressor. Architecture
exactly matches MIP Section 4.3 and the CarSpeedNet paper (arXiv:2401.07468).

Usage (from repo root):
    python models/channel_b_velocity/train.py --config models/channel_b_velocity/config.yaml

Target: velocity RMSE 1.5-2.0 m/s on held-out routes (Section 4.3).
This is achievable: CarSpeedNet's own paper reports 1.8 m/s at this
exact architecture and window size. If you land significantly above
2.0 m/s, suspect the data/preprocessing first (windowing, normalization,
unit conversion), not the model - the architecture is a known-working
reproduction.
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.channel_b_velocity.dataset import ChannelBVelocityDataset
from models.channel_b_velocity.model import ChannelBVelocityNet


class ChannelBModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ChannelBVelocityNet()
        self.loss_fn = nn.L1Loss()  # MAE, per Section 4.3 (deliberately NOT Huber like Channel A)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(torch.mean((pred - y) ** 2))  # the real Section 4.3 target metric
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse_mps", rmse, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams["optimizer"]["lr"])
        # Cosine annealing per Section 4.3 (different from Channel A's ReduceLROnPlateau).
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams["train"]["epochs"])
        return {"optimizer": opt, "lr_scheduler": sched}


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    stats_path = "data/processed/channel_b_velocity/norm_stats.json"
    train_ds = ChannelBVelocityDataset("data/processed/channel_b_velocity", "train", stats_path)
    val_ds = ChannelBVelocityDataset("data/processed/channel_b_velocity", "val", stats_path)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], num_workers=0, pin_memory=True
    )

    module = ChannelBModule(cfg)

    # Save the best checkpoint by val_loss, matching Channel A's training convention.
    # (Channel B's original train.py was missing ModelCheckpoint - it only kept the
    # last epoch, which may not be the best given early stopping.)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-val_loss={val_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["train"]["early_stopping_patience"],
            ),
            checkpoint_cb,
        ],
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"best_checkpoint={checkpoint_cb.best_model_path}")
    print(f"best_val_loss={checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/channel_b_velocity/config.yaml")
    args = parser.parse_args()
    main(args.config)
