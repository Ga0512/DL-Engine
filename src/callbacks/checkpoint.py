import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from omegaconf import DictConfig

from src.utils.s3 import upload_file


def build_checkpoint_callback(cfg: DictConfig) -> ModelCheckpoint:
    os.makedirs(cfg.checkpoint.dirpath, exist_ok=True)
    return ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename="{epoch:02d}-{step}-{val/loss:.4f}",
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )


class S3CheckpointUploader(Callback):
    """
    After each checkpoint save, upload the best checkpoint to S3.

    Usage: add to trainer callbacks when you want checkpoints persisted
    outside the ephemeral RunPod pod storage.
    """

    def __init__(self, bucket: str, prefix: str = "checkpoints"):
        self.bucket = bucket
        self.prefix = prefix

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        ckpt_cb = trainer.checkpoint_callback
        if ckpt_cb and ckpt_cb.best_model_path:
            local_path = ckpt_cb.best_model_path
            filename = os.path.basename(local_path)
            s3_key = f"{self.prefix}/{filename}"
            upload_file(local_path, self.bucket, s3_key)
