import boto3
import pytest
from moto import mock_aws

from src.utils.s3 import download_file, list_shards, upload_file


def test_upload_then_download(fake_s3, tmp_path):
    local = tmp_path / "checkpoint.ckpt"
    local.write_bytes(b"fake model weights")

    upload_file(str(local), "test-bucket", "ckpts/checkpoint.ckpt")

    dest = tmp_path / "restored.ckpt"
    download_file("test-bucket", "ckpts/checkpoint.ckpt", str(dest))

    assert dest.read_bytes() == b"fake model weights"


def test_upload_creates_parent_dirs_on_download(fake_s3, tmp_path):
    (tmp_path / "source.bin").write_bytes(b"data")
    upload_file(str(tmp_path / "source.bin"), "test-bucket", "a/b/c/file.bin")

    dest = tmp_path / "deep" / "nested" / "file.bin"
    download_file("test-bucket", "a/b/c/file.bin", str(dest))

    assert dest.exists()


def test_list_shards_only_returns_tars(fake_s3):
    for i in range(5):
        fake_s3.put_object(Bucket="test-bucket", Key=f"train/train-{i:04d}.tar", Body=b"")
    fake_s3.put_object(Bucket="test-bucket", Key="train/metadata.json", Body=b"")
    fake_s3.put_object(Bucket="test-bucket", Key="train/README.txt",    Body=b"")

    shards = list_shards("test-bucket", "train/")

    assert len(shards) == 5
    assert all(s.endswith(".tar") for s in shards)


def test_list_shards_is_sorted(fake_s3):
    # Upload in reverse order
    for i in reversed(range(8)):
        fake_s3.put_object(Bucket="test-bucket", Key=f"data/shard-{i:04d}.tar", Body=b"")

    shards = list_shards("test-bucket", "data/")

    assert shards == sorted(shards)


def test_list_shards_empty_prefix(fake_s3):
    shards = list_shards("test-bucket", "nonexistent/")
    assert shards == []
