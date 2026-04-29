import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from src.losses.focal_dice import FocalDiceLoss


class SegmentationSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        self.loss_fn = FocalDiceLoss(
            num_classes=cfg.model.num_classes,
            gamma=cfg.training.get("focal_gamma", 1.0),
            focal_weight=cfg.training.get("focal_weight", 1.0),
            dice_weight=cfg.training.get("dice_weight", 1.0),
            ignore_index=cfg.training.get("ignore_index", 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        miou = self._miou(logits.argmax(dim=1), masks)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/miou", miou, prog_bar=True, sync_dist=True)
        return loss

    def _miou(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = self.cfg.model.num_classes
        ignore = self.cfg.training.get("ignore_index", 0)
        p = preds.view(-1)
        t = targets.view(-1)
        valid = t != ignore
        p, t = p[valid], t[valid]
        ious = []
        for c in range(num_classes):
            if c == ignore:
                continue
            tp = ((p == c) & (t == c)).sum().float()
            fp = ((p == c) & (t != c)).sum().float()
            fn = ((p != c) & (t == c)).sum().float()
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)
        return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=preds.device)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt_cfg = self.cfg.optimizer
        sch_cfg = self.cfg.scheduler

        if opt_cfg.name == "adamw":
            optimizer = AdamW(self.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
        elif opt_cfg.name == "sgd":
            optimizer = SGD(self.parameters(), lr=opt_cfg.lr, momentum=0.9,
                            weight_decay=opt_cfg.weight_decay, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

        if sch_cfg.name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=opt_cfg.lr * 1e-2)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        if sch_cfg.name == "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=opt_cfg.lr,
                                   total_steps=self.trainer.estimated_stepping_batches, pct_start=0.1)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer
