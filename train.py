import sys
import wandb
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.model import build_model
from src.loaders.webdataset_loader import WebDatasetDataModule
from src.systems.lightning_module import ClassificationSystem
from src.callbacks.checkpoint import build_checkpoint_callback


def train(config_path: str = "configs/train.yaml") -> None:
    cfg = OmegaConf.load(config_path)

    pl.seed_everything(42, workers=True)

    model = build_model(cfg)
    system = ClassificationSystem(model, cfg)
    datamodule = WebDatasetDataModule(cfg)

    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity") or None,
        tags=list(cfg.wandb.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        logger=logger,
        callbacks=[
            build_checkpoint_callback(cfg),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=50,
        # IterableDataset (WebDataset) requires val_check_interval to be 1.0 or an int.
        # Use an int (number of steps) to validate N times per epoch:
        #   steps_per_epoch // 4  = 4 validations per epoch
        val_check_interval=cfg.data.samples_per_epoch // cfg.data.batch_size // 4 or 1,
        enable_progress_bar=True,
    )

    trainer.fit(system, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml"
    train(config_path)
