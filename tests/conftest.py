import io
import os
import sys
import tarfile

import boto3
import pytest
import torch
from moto import mock_aws
from omegaconf import OmegaConf
from PIL import Image

# Allow "data/prepare_dataset.py" to be imported as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


# ---------------------------------------------------------------------------
# AWS / S3 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aws_env(monkeypatch):
    """Set fake AWS credentials so boto3 never hits real AWS."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)


@pytest.fixture
def fake_s3(aws_env):
    """
    Yield a real boto3 S3 client backed by moto (fully in-memory, no network).
    Creates a 'test-bucket' ready to use.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-bucket")
        yield client


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

def _make_tar_shard(path: str, n_samples: int = 10, n_classes: int = 3) -> None:
    """Write a valid WebDataset .tar shard with JPEG + .cls pairs."""
    with tarfile.open(path, "w") as tar:
        for i in range(n_samples):
            key = f"sample-{i:06d}"

            img = Image.new("RGB", (64, 64), color=(i * 25 % 255, 80, 40))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            for name, data in [
                (f"{key}.jpg", img_bytes),
                (f"{key}.cls", str(i % n_classes).encode()),
            ]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def shard_path(tmp_path, monkeypatch) -> str:
    """
    Single .tar shard returned as a bare filename (relative path).

    WebDataset on Windows can't open absolute paths: 'C:\\...' is parsed as
    scheme 'C' and 'file:///C:/...' produces '/C:/...' which open() rejects.
    Using a relative name avoids both issues — monkeypatch restores cwd after
    each test automatically.
    """
    path = tmp_path / "train-0000.tar"
    _make_tar_shard(str(path), n_samples=10, n_classes=3)
    monkeypatch.chdir(tmp_path)
    return "train-0000.tar"


@pytest.fixture
def shard_in_s3(fake_s3, tmp_path) -> tuple[str, str]:
    """Same shard uploaded to the fake S3. Returns (bucket, key)."""
    path = str(tmp_path / "train-0000.tar")
    _make_tar_shard(path)
    fake_s3.upload_file(path, "test-bucket", "dataset/train-0000.tar")
    return "test-bucket", "dataset/train-0000.tar"


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create({
        "model": {
            "name": "resnet18",     # tiny, fast to instantiate
            "pretrained": False,    # no internet download in tests
            "num_classes": 3,
        },
        "data": {
            "train_shards": "dummy",
            "val_shards": "dummy",
            "batch_size": 4,
            "num_workers": 0,       # 0 workers = no multiprocessing issues in tests
            "image_size": 64,
            "samples_per_epoch": 8,
        },
        "training": {
            "max_epochs": 1,
            "precision": "32",      # no AMP — CPU-safe
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
        },
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "scheduler": {"name": "cosine"},
        "checkpoint": {
            "dirpath": str(tmp_path / "checkpoints"),
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 1,
        },
        "wandb": {"project": "test", "entity": None, "tags": []},
    })
