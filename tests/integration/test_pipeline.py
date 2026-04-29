"""
Integration test: full training pipeline using a local folder as fake S3 bucket.

tests/assets/bucket/
    train/train-0000.tar  ...   <- simulates s3://bucket/train/*.tar
    val/val-0000.tar            <- simulates s3://bucket/val/*.tar

No moto, no boto3, no network. _expand_to_pipe_urls is patched to point
at the local bucket folder instead of real S3.
"""

from pathlib import Path

import pytorch_lightning as pl
import pytest

from src.loaders.webdataset_loader import WebDatasetDataModule
from src.models.model import build_model
from src.systems.lightning_module import ClassificationSystem

BUCKET = Path(__file__).parent.parent / "assets" / "bucket"
CLASSES = ["bird", "cat", "dog"]   # must match make_assets.py


@pytest.fixture
def bucket_cfg(cfg):
    from omegaconf import OmegaConf
    # Pass local paths directly — _expand_to_pipe_urls detects non-s3:// patterns
    # and converts them to localfile:// URLs (no subprocess, no BrokenPipe).
    return OmegaConf.merge(cfg, {
        "model": {"num_classes": len(CLASSES)},
        "data": {
            "train_shards": str(BUCKET / "train" / "train-{0000..0002}.tar"),
            "val_shards":   str(BUCKET / "val"   / "val-{0000..0000}.tar"),
            "samples_per_epoch": 8,
            "batch_size": 4,
        },
    })


# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BUCKET.exists(),
    reason="Run 'python tests/make_assets.py' to generate the fake bucket",
)
def test_bucket_has_shards():
    """Sanity check: the fake bucket has .tar files in train/ and val/."""
    train_shards = list((BUCKET / "train").glob("*.tar"))
    val_shards   = list((BUCKET / "val").glob("*.tar"))
    assert train_shards, f"No shards in {BUCKET / 'train'}"
    assert val_shards,   f"No shards in {BUCKET / 'val'}"


@pytest.mark.skipif(
    not BUCKET.exists(),
    reason="Run 'python tests/make_assets.py' to generate the fake bucket",
)
def test_full_training_pipeline(bucket_cfg):
    """
    Full pipeline using the local bucket folder — no mocks.

    _expand_to_pipe_urls detects non-s3:// paths and uses the localfile://
    gopen handler, so WebDataset reads directly from disk without a subprocess.
    """
    dm     = WebDatasetDataModule(bucket_cfg)
    model  = build_model(bucket_cfg)
    system = ClassificationSystem(model, bucket_cfg)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(system, datamodule=dm)
