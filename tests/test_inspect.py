"""
Tests for data/prepare_dataset.py (the S3 inspection utilities).
Imported via sys.path set in conftest.py.
"""

import boto3
import pytest
from moto import mock_aws

import prepare_dataset as inspect_mod


def _populate_shards(client, bucket: str, prefix: str, n: int) -> None:
    for i in range(n):
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/train-{i:04d}.tar",
            Body=b"x" * 100,
        )


# ---------------------------------------------------------------------------
# list_shards
# ---------------------------------------------------------------------------

def test_list_shards_returns_only_tars(fake_s3):
    _populate_shards(fake_s3, "test-bucket", "data", 6)
    fake_s3.put_object(Bucket="test-bucket", Key="data/metadata.json", Body=b"")

    shards = inspect_mod.list_shards("test-bucket", "data/")

    assert len(shards) == 6
    assert all(k.endswith(".tar") for k in shards)


def test_list_shards_sorted(fake_s3):
    _populate_shards(fake_s3, "test-bucket", "data", 8)

    shards = inspect_mod.list_shards("test-bucket", "data/")

    assert shards == sorted(shards)


def test_list_shards_empty(fake_s3):
    shards = inspect_mod.list_shards("test-bucket", "no-such-prefix/")
    assert shards == []


# ---------------------------------------------------------------------------
# build_pattern
# ---------------------------------------------------------------------------

def test_build_pattern_brace_notation():
    shards = [f"train/train-{i:04d}.tar" for i in range(8)]
    pattern = inspect_mod.build_pattern("my-bucket", "train", shards)

    assert "s3://my-bucket" in pattern
    assert "{0000..0007}" in pattern


def test_build_pattern_single_shard():
    shards = ["train/train-0000.tar"]
    pattern = inspect_mod.build_pattern("my-bucket", "train", shards)

    assert "{0000..0000}" in pattern


def test_build_pattern_non_sequential_falls_back_to_list():
    # Filenames with no numeric suffix — brace pattern not possible
    shards = ["data/alpha.tar", "data/beta.tar", "data/gamma.tar"]
    pattern = inspect_mod.build_pattern("my-bucket", "data", shards)

    # Fallback: one URL per line
    lines = pattern.strip().splitlines()
    assert len(lines) == 3
    assert all("s3://my-bucket" in line for line in lines)


# ---------------------------------------------------------------------------
# inspect (integration: list + size check + pattern)
# ---------------------------------------------------------------------------

def test_inspect_prints_pattern(fake_s3, capsys):
    _populate_shards(fake_s3, "test-bucket", "train", 4)

    inspect_mod.inspect("test-bucket", "train/", peek=False)

    captured = capsys.readouterr()
    assert "train_shards" in captured.out
    assert "{0000..0003}" in captured.out
    assert "4" in captured.out          # shard count


def test_inspect_no_shards_warns(fake_s3, capsys):
    inspect_mod.inspect("test-bucket", "empty/", peek=False)

    captured = capsys.readouterr()
    assert "No .tar" in captured.out
