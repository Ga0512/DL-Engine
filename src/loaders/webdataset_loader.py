import sys
import urllib.parse

import torch
import webdataset as wds
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import braceexpand


# ---------------------------------------------------------------------------
# Register a "localfile://" scheme so WebDataset can open local .tar files
# without spawning a subprocess (avoids BrokenPipeError on Windows).
#
# webdataset.__init__ re-exports gopen as a function, shadowing the module,
# so we access the real module via sys.modules after import.
# ---------------------------------------------------------------------------

def _localfile_handler(url: str, mode: str = "rb", bufsize: int = 8192, **kw):
    path = urllib.parse.urlparse(url).path
    # urlparse("localfile:///C:/foo").path == "/C:/foo" on Windows — strip leading /
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return open(path, mode)

sys.modules["webdataset.gopen"].gopen_schemes["localfile"] = _localfile_handler


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# URL expansion
# ---------------------------------------------------------------------------

def _expand_to_pipe_urls(shard_pattern: str) -> list[str]:
    """
    Expand brace notation and return URLs WebDataset can stream.

    s3://bucket/train-{0000..0127}.tar  → pipe:aws s3 cp s3://... -
    tests/assets/bucket/train-{...}.tar → localfile:///abs/path/train-0000.tar
    """
    # braceexpand treats \ as an escape character — normalize to / first.
    shard_pattern = shard_pattern.replace("\\", "/")
    paths = list(braceexpand.braceexpand(shard_pattern))

    if paths[0].startswith("s3://"):
        return [f"pipe:aws s3 cp {p} -" for p in paths]

    # Local paths: use the registered localfile:// handler (no subprocess, no BrokenPipe)
    import os
    result = []
    for p in paths:
        abs_path = os.path.abspath(p).replace("\\", "/")
        result.append(f"localfile:///{abs_path}")
    return result


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class WebDatasetDataModule(pl.LightningDataModule):
    """
    DataModule that streams WebDataset shards from S3 via pipe: commands,
    or from a local folder (for testing) via the localfile:// handler.

    Shard format expected inside each .tar:
        {key}.jpg  — JPEG image
        {key}.cls  — integer class label as plain text

    For segmentation, swap .cls → .png (mask).
    For multi-label or regression, swap .cls → .json.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def _make_dataset(self, shard_pattern: str, is_train: bool) -> wds.WebDataset:
        urls = _expand_to_pipe_urls(shard_pattern)
        transform = _train_transform(self.cfg.data.image_size) if is_train else _val_transform(self.cfg.data.image_size)

        dataset = (
            wds.WebDataset(
                urls,
                resampled=is_train,
                shardshuffle=500 if is_train else False,
                nodesplitter=wds.split_by_node,
            )
            .shuffle(2000 if is_train else 0)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(transform, lambda x: torch.tensor(int(x), dtype=torch.long))
            .batched(self.cfg.data.batch_size, partial=not is_train)
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataset = self._make_dataset(self.cfg.data.train_shards, is_train=True)
        steps = self.cfg.data.samples_per_epoch // self.cfg.data.batch_size
        dataset = dataset.with_epoch(steps)

        nw = self.cfg.data.num_workers
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=4 if nw > 0 else None,
            persistent_workers=nw > 0,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._make_dataset(self.cfg.data.val_shards, is_train=False)

        nw = max(self.cfg.data.num_workers // 2, 2) if self.cfg.data.num_workers > 0 else 0
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=2 if nw > 0 else None,
            persistent_workers=nw > 0,
        )
