"""
Generate synthetic fixture images and pack them into WebDataset shards.

Run once before the integration tests:
    python tests/make_assets.py

Output structure (simulates an S3 bucket locally):
    tests/assets/bucket/
        train/
            train-0000.tar
            train-0001.tar
        val/
            val-0000.tar
"""

import io
import random
import tarfile
from pathlib import Path

from PIL import Image

CLASSES = {"cat": 0, "dog": 1, "bird": 2}
N_PER_CLASS = 8
SHARD_SIZE = 8
BASE_COLORS = {"cat": (220, 80, 60), "dog": (60, 160, 220), "bird": (80, 200, 100)}

BUCKET_DIR = Path(__file__).parent / "assets" / "bucket"


def make_image(base_color: tuple[int, int, int], seed: int) -> bytes:
    random.seed(seed)
    r = max(0, min(255, base_color[0] + random.randint(-30, 30)))
    g = max(0, min(255, base_color[1] + random.randint(-30, 30)))
    b = max(0, min(255, base_color[2] + random.randint(-30, 30)))
    img = Image.new("RGB", (128, 128), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def write_shard(path: Path, samples: list[tuple[str, bytes, int]]) -> None:
    with tarfile.open(path, "w") as tar:
        for key, img_bytes, label in samples:
            for name, data in [
                (f"{key}.jpg", img_bytes),
                (f"{key}.cls", str(label).encode()),
            ]:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def pack(split: str, samples: list[tuple[str, bytes, int]]) -> list[Path]:
    out_dir = BUCKET_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    for shard_idx, start in enumerate(range(0, len(samples), SHARD_SIZE)):
        chunk = samples[start : start + SHARD_SIZE]
        path = out_dir / f"{split}-{shard_idx:04d}.tar"
        write_shard(path, chunk)
        shard_paths.append(path)
    return shard_paths


def main() -> None:
    all_samples = [
        (f"sample-{cls}-{i:03d}", make_image(BASE_COLORS[cls], seed=i + idx * 100), label)
        for idx, (cls, label) in enumerate(CLASSES.items())
        for i in range(N_PER_CLASS)
    ]

    # 80/20 train/val split
    split_at = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_at]
    val_samples   = all_samples[split_at:]

    train_shards = pack("train", train_samples)
    val_shards   = pack("val",   val_samples)

    print(f"Bucket -> {BUCKET_DIR}")
    print(f"  train/  {len(train_shards)} shards  ({len(train_samples)} samples)")
    print(f"  val/    {len(val_shards)} shards  ({len(val_samples)} samples)")
    print(f"\nShard pattern for configs/train.yaml:")
    print(f'  train_shards: "s3://my-bucket/train/train-{{0000..{len(train_shards)-1:04d}}}.tar"')
    print(f'  val_shards:   "s3://my-bucket/val/val-{{0000..{len(val_shards)-1:04d}}}.tar"')


if __name__ == "__main__":
    main()
