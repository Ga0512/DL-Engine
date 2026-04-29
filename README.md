# DL Engine

A PyTorch Lightning training template for GPU pods on RunPod. Data streams from S3 via WebDataset — no local copy of the dataset is ever needed. Experiments log to W&B. Designed to be task-agnostic: swap one file to go from classification to segmentation, HuggingFace, or anything else.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [Quick Start (local)](#quick-start-local)
4. [Training on RunPod](#training-on-runpod)
5. [Dataset Format](#dataset-format)
6. [Inspecting Shards in S3](#inspecting-shards-in-s3)
7. [Configuration Reference](#configuration-reference)
8. [Adapting for Other Tasks](#adapting-for-other-tasks)
9. [Checkpointing](#checkpointing)
10. [Tests](#tests)
11. [Docker](#docker)
12. [Inference Endpoint (optional)](#inference-endpoint-optional)

---

## Project Structure

```
train.py                          # Entrypoint — wires everything together
configs/
    train.yaml                    # Production config (S3 shards)
    train_local.yaml              # Local dev config (fake bucket)

src/
    models/model.py               # build_model(cfg) → nn.Module via timm
    loaders/webdataset_loader.py  # WebDatasetDataModule — streams from S3
    systems/lightning_module.py   # ClassificationSystem — loss, optimizer, scheduler
    callbacks/checkpoint.py       # ModelCheckpoint + optional S3 upload
    utils/s3.py                   # Thin boto3 wrappers

data/
    prepare_dataset.py            # Inspect shards already in S3

tests/
    make_assets.py                # Generate fake local bucket (run once)
    assets/bucket/                # Fake bucket used in tests and local training
    test_loader.py
    test_system.py
    test_checkpoint.py
    test_s3.py
    test_inspect.py
    integration/
        test_pipeline.py          # Full train/val loop against local bucket

runpod_handler.py                 # Serverless inference endpoint (optional)
Dockerfile
requirements.txt
requirements-dev.txt
```

---

## How It Works

```
S3 (.tar shards)
  └─ pipe:aws s3 cp s3://bucket/train-0000.tar -   # streamed, no disk write
       └─ wds.WebDataset(resampled=True)            # infinite stream for training
            └─ .decode("pil")                       # auto-decode JPEG → PIL
            └─ .to_tuple("jpg", "cls")              # unpack by extension
            └─ transform → batch → GPU
```

Each `.tar` shard contains paired files:

```
sample-001.jpg   ← JPEG image
sample-001.cls   ← integer class label as plain text ("2")
```

WebDataset reads the files as key-value pairs grouped by stem (`sample-001`). The pipeline decodes and maps them to tensors before batching.

Training uses `resampled=True`, which turns the dataset into an infinite stream. `with_epoch(steps)` defines how many batches constitute one epoch — this lets Lightning's validation, checkpointing, and LR scheduling work normally even though the data never ends.

---

## Quick Start (local)

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed for tests
```

### 3. Generate the fake local bucket

```bash
python tests/make_assets.py
```

This creates `tests/assets/bucket/` with synthetic JPEG images (cat/dog/bird) packed into WebDataset shards. It simulates an S3 bucket locally — no AWS credentials needed.

```
tests/assets/bucket/
    train/
        train-0000.tar
        train-0001.tar
        train-0002.tar
    val/
        val-0000.tar
```

### 4. Train locally against the fake bucket

`configs/train_local.yaml` points directly at the folder you just created:

```yaml
data:
  train_shards: "tests/assets/bucket/train/train-{0000..0002}.tar"
  val_shards:   "tests/assets/bucket/val/val-{0000..0000}.tar"
```

This is a **real training run** — same code path as production, same DataModule, same Lightning Trainer. The only difference is that the shards come from disk instead of S3. No AWS credentials needed.

```bash
WANDB_MODE=disabled python train.py configs/train_local.yaml
```

You should see Lightning's progress bar and `train/loss` + `val/loss` logged each epoch. A successful run confirms that your data pipeline, model, optimizer, and checkpoint callback all work before you touch a GPU pod.

If you do have a W&B account, drop `WANDB_MODE=disabled` and set your key:

```bash
WANDB_API_KEY=<your-key> python train.py configs/train_local.yaml
```

---

## Training on RunPod

### 1. Set environment variables on the pod

In the RunPod dashboard, under **Environment Variables**:

```
AWS_ACCESS_KEY_ID       = <your key>
AWS_SECRET_ACCESS_KEY   = <your secret>
AWS_DEFAULT_REGION      = us-east-1
WANDB_API_KEY           = <your key>
```

### 2. SSH into the pod and run

```bash
git clone <your-repo>
cd dl-engine
pip install -r requirements.txt
python train.py configs/train.yaml
```

That's it. No dataset download step — data streams directly from S3 into GPU memory.

### Choosing the right precision

| GPU          | `training.precision` |
|--------------|----------------------|
| V100 / RTX   | `"16-mixed"`         |
| A100 / H100  | `"bf16-mixed"`       |
| CPU / debug  | `"32"`               |

---

## Dataset Format

Your shards must be WebDataset `.tar` files. Each sample inside the tar is a group of files sharing the same stem:

| Task           | Files per sample                          |
|----------------|-------------------------------------------|
| Classification | `{key}.jpg` + `{key}.cls`                 |
| Segmentation   | `{key}.jpg` + `{key}.png` (mask)          |
| JSON metadata  | `{key}.jpg` + `{key}.json`                |
| Multi-modal    | `{key}.jpg` + `{key}.txt` + `{key}.json`  |

The `.cls` file contains a plain-text integer, e.g. `"2"`. Adjust `.to_tuple(...)` in `webdataset_loader.py` to match your extensions.

#### Shard pattern in the config

Use brace expansion to describe a range of shards:

```yaml
train_shards: "s3://my-bucket/train/train-{0000..0127}.tar"
val_shards:   "s3://my-bucket/val/val-{0000..0015}.tar"
```

This expands to 128 train shards and 16 val shards. The pattern is printed for you by `prepare_dataset.py`.

---

## Inspecting Shards in S3

Before training, use `prepare_dataset.py` to verify your bucket layout:

```bash
# List shards and get the brace-expanded pattern to paste into train.yaml
python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train

# Also peek inside the first shard to see what extensions are inside
python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train --peek
```

Example output:

```
Listing shards in s3://my-bucket/dataset/train ...

  Shards found : 128
  Total size   : 24.3 GB
  First shard  : dataset/train/train-0000.tar
  Last shard   : dataset/train/train-0127.tar

  Pattern for configs/train.yaml:
    train_shards: "s3://my-bucket/dataset/train/train-{0000..0127}.tar"

Peeking into s3://my-bucket/dataset/train/train-0000.tar
  Found 3 sample(s):
    'sample-001'  →  ['jpg', 'cls']
    'sample-002'  →  ['jpg', 'cls']
    'sample-003'  →  ['jpg', 'cls']

  Extensions in shard: ['cls', 'jpg']
  → In webdataset_loader.py, use: .to_tuple('cls', 'jpg')
```

---

## Configuration Reference

```yaml
model:
  name: resnet50        # Any timm model name (see below)
  pretrained: true      # ImageNet weights from timm
  num_classes: 10       # Output classes

data:
  train_shards: "s3://bucket/train/train-{0000..0127}.tar"
  val_shards:   "s3://bucket/val/val-{0000..0015}.tar"
  batch_size: 64
  num_workers: 8        # Rule of thumb: 4 per GPU, max 16
  image_size: 224
  samples_per_epoch: 1_000_000  # Epoch length for infinite streaming

training:
  max_epochs: 50
  precision: "16-mixed"         # "bf16-mixed" on A100/H100
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1    # Effective batch = batch_size * this

optimizer:
  name: adamw           # "adamw" or "sgd"
  lr: 1.0e-3
  weight_decay: 1.0e-4

scheduler:
  name: cosine          # "cosine" or "onecycle"

checkpoint:
  dirpath: "./checkpoints"
  monitor: "val/loss"
  mode: "min"
  save_top_k: 3         # Keep 3 best + last.ckpt

wandb:
  project: "dl-engine"
  entity: null          # W&B username or team name
  tags: []
```

### Common timm model names

```
resnet18, resnet50, resnet101
vit_base_patch16_224, vit_large_patch16_224
convnext_base, convnext_large
efficientnet_b4, efficientnet_b7
swin_base_patch4_window7_224
```

Browse all models: `timm.list_models(pretrained=True)`

---

## Adapting for Other Tasks

The only files you need to touch are `model.py` and `lightning_module.py`.

### Segmentation (e.g. UNet with smp)

**`src/models/model.py`**
```python
import segmentation_models_pytorch as smp

def build_model(cfg):
    return smp.Unet(
        encoder_name=cfg.model.name,    # e.g. "resnet50"
        encoder_weights="imagenet" if cfg.model.pretrained else None,
        classes=cfg.model.num_classes,
    )
```

**`src/systems/lightning_module.py`** — change the loss:
```python
loss = smp.losses.DiceLoss(mode="multiclass")(logits, labels)
```

**`src/loaders/webdataset_loader.py`** — change the extensions:
```python
.to_tuple("jpg", "png")                                # image + mask
.map_tuple(transform, mask_transform)
```

### HuggingFace / VLM

**`src/models/model.py`**
```python
from transformers import AutoModel

def build_model(cfg):
    return AutoModel.from_pretrained(cfg.model.name)
```

**`src/systems/lightning_module.py`** — override `training_step` to unpack the batch the way HuggingFace models expect:
```python
def training_step(self, batch, batch_idx):
    outputs = self.model(**batch)
    self.log("train/loss", outputs.loss)
    return outputs.loss
```

### Regression

**`src/systems/lightning_module.py`** — swap the loss and remove accuracy:
```python
loss = F.mse_loss(logits.squeeze(), labels.float())
self.log(f"{stage}/loss", loss)
return loss
```

### Multi-GPU (DDP)

No code changes required. Lightning + `split_by_node` in the dataloader handles it automatically. Just launch with:

```bash
python train.py configs/train.yaml  # Lightning detects all GPUs on the pod
```

Or explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py configs/train.yaml
```

---

## Checkpointing

Checkpoints are saved locally to `checkpoint.dirpath`. The best K checkpoints (by `val/loss`) and `last.ckpt` are always kept.

### Persist checkpoints to S3

To survive pod restarts, add `S3CheckpointUploader` to the trainer callbacks in `train.py`:

```python
from src.callbacks.checkpoint import S3CheckpointUploader

trainer = pl.Trainer(
    ...
    callbacks=[
        build_checkpoint_callback(cfg),
        LearningRateMonitor(logging_interval="step"),
        S3CheckpointUploader(bucket="my-bucket", prefix="checkpoints/run-01"),
    ],
)
```

Every time a new best checkpoint is saved locally, it is automatically uploaded to `s3://my-bucket/checkpoints/run-01/`.

### Resume training from a checkpoint

```python
trainer.fit(system, datamodule=datamodule, ckpt_path="checkpoints/last.ckpt")
```

---

## Tests

### Generate the fake bucket (once)

```bash
python tests/make_assets.py
```

### Run all tests

```bash
pytest tests/
```

### What each test file covers

| File                            | What it tests                                              |
|---------------------------------|------------------------------------------------------------|
| `tests/integration/test_pipeline.py` | Full train + val loop against local bucket (no mocks) |
| `tests/test_loader.py`          | Transforms, WebDataset reading, batching                  |
| `tests/test_system.py`          | Forward pass, loss, optimizer, scheduler                  |
| `tests/test_checkpoint.py`      | ModelCheckpoint creation, S3 upload callback              |
| `tests/test_s3.py`              | S3 upload, download, list_shards (moto fake S3)           |
| `tests/test_inspect.py`         | Shard listing, brace-pattern generation                   |

---

## Docker

```bash
# Build
docker build -t dl-engine .

# Run with GPU and credentials from .env
docker run --gpus all --env-file .env dl-engine
```

The `.env` file should contain:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
WANDB_API_KEY=...
```

---

## Inference Endpoint (optional)

`runpod_handler.py` is a RunPod Serverless handler — only needed if you want a pay-per-request inference API that scales to zero. It is **not** used during training.

If you only train, you can delete this file.

To deploy:

1. Build and push the Docker image to a registry
2. Create a RunPod Serverless endpoint pointing to that image
3. Set `CONFIG_PATH` and `CHECKPOINT_PATH` as environment variables on the endpoint

Request format:

```json
{
    "input": {
        "image_b64": "<base64-encoded JPEG>"
    }
}
```

Response:

```json
{
    "class_id": 3,
    "confidence": 0.97
}
```
