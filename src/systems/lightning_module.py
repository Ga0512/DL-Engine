import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


class ClassificationSystem(pl.LightningModule):
    """
    Training system for image classification.

    To adapt for other tasks:
      Segmentation  → swap F.cross_entropy with your seg loss, adjust _shared_step metrics
      Regression    → swap cross_entropy with F.mse_loss, remove accuracy metric
      VLM/Multimodal → override training_step, pull multiple keys from batch
    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc",  acc,  prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt_cfg = self.cfg.optimizer
        sch_cfg = self.cfg.scheduler

        if opt_cfg.name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_cfg.name == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=opt_cfg.lr,
                momentum=0.9,
                weight_decay=opt_cfg.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

        if sch_cfg.name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=opt_cfg.lr * 1e-2,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if sch_cfg.name == "onecycle":
            # OneCycleLR needs total_steps — only valid after trainer is attached
            scheduler = OneCycleLR(
                optimizer,
                max_lr=opt_cfg.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
