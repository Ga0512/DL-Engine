import torch
import pytorch_lightning as pl
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import build_model
from src.systems.lightning_module import ClassificationSystem


class FakeDataModule(pl.LightningDataModule):
    """
    Synthetic data — no S3, no WebDataset, no disk.
    Used to test the training system in isolation.
    """

    def __init__(self, n_samples: int, num_classes: int, image_size: int, batch_size: int):
        super().__init__()
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size

    def _loader(self) -> DataLoader:
        images = torch.randn(self.n_samples, 3, self.image_size, self.image_size)
        labels = torch.randint(0, self.num_classes, (self.n_samples,))
        return DataLoader(TensorDataset(images, labels), batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_build_model_output_shape(cfg):
    model = build_model(cfg)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, cfg.model.num_classes)


def test_build_model_wrong_name_raises(cfg):
    from omegaconf import OmegaConf
    bad_cfg = OmegaConf.merge(cfg, {"model": {"name": "not_a_real_model_xyz"}})
    with pytest.raises(Exception):
        build_model(bad_cfg)


# ---------------------------------------------------------------------------
# System forward / loss
# ---------------------------------------------------------------------------

def test_system_forward_pass(cfg):
    model = build_model(cfg)
    system = ClassificationSystem(model, cfg)

    x = torch.randn(4, 3, 64, 64)
    logits = system(x)
    assert logits.shape == (4, cfg.model.num_classes)


def test_training_step_returns_scalar_loss(cfg):
    model = build_model(cfg)
    system = ClassificationSystem(model, cfg)

    images = torch.randn(4, 3, 64, 64)
    labels = torch.randint(0, cfg.model.num_classes, (4,))
    batch = (images, labels)

    # training_step needs self.trainer for logging — attach a minimal one
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.strategy.connect(system)

    loss = system.training_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Full training loop (fast_dev_run = 1 batch train + 1 batch val)
# ---------------------------------------------------------------------------

def test_full_training_loop(cfg):
    model = build_model(cfg)
    system = ClassificationSystem(model, cfg)
    datamodule = FakeDataModule(
        n_samples=16,
        num_classes=cfg.model.num_classes,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,              # 1 train + 1 val batch, then stop
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(system, datamodule=datamodule)


def test_sgd_optimizer(cfg):
    from omegaconf import OmegaConf
    sgd_cfg = OmegaConf.merge(cfg, {"optimizer": {"name": "sgd"}})

    model = build_model(sgd_cfg)
    system = ClassificationSystem(model, sgd_cfg)
    datamodule = FakeDataModule(16, sgd_cfg.model.num_classes, sgd_cfg.data.image_size, 4)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(system, datamodule=datamodule)


def test_onecycle_scheduler(cfg):
    from omegaconf import OmegaConf
    oc_cfg = OmegaConf.merge(cfg, {"scheduler": {"name": "onecycle"}})

    model = build_model(oc_cfg)
    system = ClassificationSystem(model, oc_cfg)
    datamodule = FakeDataModule(16, oc_cfg.model.num_classes, oc_cfg.data.image_size, 4)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(system, datamodule=datamodule)
