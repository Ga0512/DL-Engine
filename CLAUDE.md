# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A practical PyTorch Lightning training template for GPU pods on RunPod. Data streams from S3 via WebDataset (no local copy). Experiments log to W&B. Designed to be task-agnostic (classification, segmentation, VLM, OCR) without enterprise overengineering.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Inspect shards already in S3 (list, count, get the pattern for train.yaml)
python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train

# Peek inside a shard to see what keys/extensions are inside the .tar
python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train --peek

# Train
python train.py                          # uses configs/train.yaml
python train.py configs/my_exp.yaml      # custom config

# Docker
docker build -t dl-engine .
docker run --gpus all --env-file .env dl-engine

# RunPod: SSH into the pod and run the same as local:
python train.py configs/train.yaml
```

## Architecture

```
train.py                    ← entrypoint: wires everything together
configs/train.yaml          ← all hyperparameters, paths, W&B config

src/models/model.py         ← build_model(cfg) → nn.Module via timm
src/loaders/webdataset_loader.py  ← WebDatasetDataModule: streams shards from S3
src/systems/lightning_module.py   ← ClassificationSystem: loss, optimizer, scheduler
src/callbacks/checkpoint.py       ← ModelCheckpoint + optional S3 upload callback
src/utils/s3.py             ← thin boto3 wrappers (upload, download, list_shards)

data/prepare_dataset.py     ← one-time script: raw images → .tar shards on S3
runpod_handler.py           ← RunPod Serverless inference handler (optional, not for training)
```

## Data flow

```
S3 (.tar shards)
  → pipe:aws s3 cp s3://... -      # shell pipe inside WebDataset — no disk write
  → wds.WebDataset(resampled=True) # infinite stream for training
  → .decode("pil")                 # auto-decode jpg → PIL
  → .to_tuple("jpg", "cls")        # unpack by file extension inside the tar
  → transform → batch → GPU
```

Shard format inside each `.tar`:
- `{key}.jpg` — JPEG image
- `{key}.cls` — integer class label as plain text

For segmentation swap `.cls` → `.png` (mask). For JSON metadata use `.json`.

## Key config fields

- `data.train_shards`: brace-expanded S3 pattern, e.g. `"s3://bucket/train-{0000..0127}.tar"`
- `data.samples_per_epoch`: controls epoch length when using `resampled=True` (infinite stream)
- `training.precision`: `"16-mixed"` for V100/RTX, `"bf16-mixed"` for A100/H100
- `model.name`: any timm model name; swap `build_model` for smp/HuggingFace for other tasks

## Extending for other tasks

| Task | What to change |
|------|---------------|
| Segmentation | `build_model` → `segmentation_models_pytorch`; `_shared_step` loss |
| HuggingFace/VLM | `build_model` → `AutoModel.from_pretrained`; `training_step` for your loss |
| Regression | `cross_entropy` → `F.mse_loss`; remove accuracy metric |
| Multi-GPU DDP | No changes — Lightning + `split_by_node` handles it automatically |

## Environment variables

Required at runtime (set in RunPod pod environment or `.env`):

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
WANDB_API_KEY
```

## RunPod Pod vs Serverless

- **Pod** (for training): rent a GPU, SSH in, run `python train.py`. No handler needed.
- **Serverless** (`runpod_handler.py`): only for inference APIs that scale to zero. Delete it if you only train.
