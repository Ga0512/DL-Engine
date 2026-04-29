import torch
import webdataset as wds
import pytest

from src.loaders.webdataset_loader import _train_transform, _val_transform
from PIL import Image


# ---------------------------------------------------------------------------
# Transform unit tests — pure CPU, no S3
# ---------------------------------------------------------------------------

def test_train_transform_shape():
    img = Image.new("RGB", (256, 256))
    tensor = _train_transform(64)(img)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32


def test_val_transform_shape():
    img = Image.new("RGB", (256, 256))
    tensor = _val_transform(64)(img)
    assert tensor.shape == (3, 64, 64)


def test_transform_normalizes_range():
    # After ImageNet normalization values should not be in [0, 1] anymore
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    tensor = _val_transform(64)(img)
    assert tensor.min().item() < 0 or tensor.max().item() > 1


# ---------------------------------------------------------------------------
# WebDataset pipeline — reads the local fake shard (no S3, no pipe:)
# ---------------------------------------------------------------------------

def test_webdataset_reads_local_shard(shard_path):
    transform = _val_transform(64)

    dataset = (
        wds.WebDataset(shard_path)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, lambda x: torch.tensor(int(x), dtype=torch.long))
        .batched(4, partial=True)
    )

    batch = next(iter(dataset))
    images, labels = batch

    assert images.shape == (4, 3, 64, 64)
    assert labels.dtype == torch.long
    assert labels.max().item() < 3     # n_classes=3 in conftest


def test_webdataset_yields_all_samples(shard_path):
    dataset = (
        wds.WebDataset(shard_path)
        .decode("pil")
        .to_tuple("jpg", "cls")
    )

    samples = list(dataset)
    assert len(samples) == 10           # n_samples=10 in conftest


def test_webdataset_batched_shapes(shard_path):
    transform = _train_transform(64)

    dataset = (
        wds.WebDataset(shard_path)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, lambda x: torch.tensor(int(x), dtype=torch.long))
        .batched(3, partial=True)
    )

    batches = list(dataset)
    total_samples = sum(b[0].shape[0] for b in batches)
    assert total_samples == 10

    # All full batches should have the right image shape
    for images, labels in batches:
        assert images.ndim == 4
        assert images.shape[1:] == (3, 64, 64)
        assert labels.ndim == 1
