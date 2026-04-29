import boto3
import pytest
from unittest.mock import MagicMock
from moto import mock_aws

from src.callbacks.checkpoint import S3CheckpointUploader, build_checkpoint_callback


def test_build_checkpoint_callback_creates_dir(cfg, tmp_path):
    callback = build_checkpoint_callback(cfg)
    assert callback.dirpath == cfg.checkpoint.dirpath
    assert callback.monitor == "val/loss"
    assert callback.mode == "min"


def test_s3_uploader_uploads_best_checkpoint(fake_s3, tmp_path):
    ckpt_file = tmp_path / "epoch=00-0.1234.ckpt"
    ckpt_file.write_bytes(b"serialized model state")

    ckpt_cb = MagicMock()
    ckpt_cb.best_model_path = str(ckpt_file)

    trainer = MagicMock()
    trainer.checkpoint_callback = ckpt_cb

    uploader = S3CheckpointUploader(bucket="test-bucket", prefix="runs/exp01")
    uploader.on_save_checkpoint(trainer, MagicMock(), {})

    response = fake_s3.get_object(Bucket="test-bucket", Key="runs/exp01/epoch=00-0.1234.ckpt")
    assert response["Body"].read() == b"serialized model state"


def test_s3_uploader_skips_when_no_best_path(fake_s3):
    """Should not raise if best_model_path is empty (first epoch not done yet)."""
    ckpt_cb = MagicMock()
    ckpt_cb.best_model_path = ""

    trainer = MagicMock()
    trainer.checkpoint_callback = ckpt_cb

    uploader = S3CheckpointUploader(bucket="test-bucket", prefix="ckpts")
    uploader.on_save_checkpoint(trainer, MagicMock(), {})   # must not raise


def test_s3_uploader_skips_when_no_checkpoint_callback(fake_s3):
    trainer = MagicMock()
    trainer.checkpoint_callback = None

    uploader = S3CheckpointUploader(bucket="test-bucket", prefix="ckpts")
    uploader.on_save_checkpoint(trainer, MagicMock(), {})   # must not raise
