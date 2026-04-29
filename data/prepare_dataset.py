"""
Inspect WebDataset shards already stored in S3.

Does three things:
  1. Lists all .tar shards under a bucket prefix
  2. Peeks inside one shard to show what keys/extensions exist
  3. Prints the brace-expanded pattern to paste into configs/train.yaml

Usage:
    python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train
    python data/prepare_dataset.py --s3_bucket my-bucket --s3_prefix dataset/train --peek
"""

import argparse
import io
import os
import subprocess
import tarfile

import boto3


def list_shards(bucket: str, prefix: str) -> list[str]:
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    shards = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar"):
                shards.append(obj["Key"])
    return sorted(shards)


def peek_shard(bucket: str, s3_key: str, max_samples: int = 3) -> None:
    """Stream the first bytes of a shard via pipe and print its member names."""
    url = f"s3://{bucket}/{s3_key}"
    print(f"\nPeeking into {url}")

    cmd = ["aws", "s3", "cp", url, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Read enough bytes to see the tar members without downloading the whole shard
    buf = io.BytesIO()
    chunk_size = 512 * 1024  # 512 KB — enough to see file headers
    seen = 0
    for chunk in proc.stdout:
        buf.write(chunk)
        seen += len(chunk)
        if seen >= chunk_size * 20:  # ~10 MB max
            break
    proc.terminate()

    buf.seek(0)
    try:
        with tarfile.open(fileobj=buf, mode="r|*") as tar:
            sample_keys: dict[str, list[str]] = {}
            for member in tar:
                if not member.name:
                    continue
                stem, _, ext = member.name.rpartition(".")
                sample_keys.setdefault(stem, []).append(ext)
                if len(sample_keys) >= max_samples:
                    break

        print(f"  Found {len(sample_keys)} sample(s) (showing up to {max_samples}):")
        for key, exts in sample_keys.items():
            print(f"    {key!r}  →  {exts}")

        all_exts = sorted({ext for exts in sample_keys.values() for ext in exts})
        print(f"\n  Extensions in shard: {all_exts}")
        print(f"  → In webdataset_loader.py, use: .to_tuple({', '.join(repr(e) for e in all_exts)})")
    except Exception as e:
        print(f"  Could not parse tar: {e}")


def build_pattern(bucket: str, prefix: str, shards: list[str]) -> str:
    """
    Generate the brace-expanded S3 pattern from the actual shard list.

    Handles both zero-padded sequences and arbitrary names.
    """
    filenames = [os.path.basename(k) for k in shards]

    # Try to detect a numeric sequence: name-0000.tar … name-0127.tar
    import re
    pattern = re.compile(r"^(.+?-)(\d+)(\.tar)$")
    matches = [pattern.match(f) for f in filenames]

    if all(matches):
        prefix_part = matches[0].group(1)
        suffix_part = matches[0].group(3)
        indices = [m.group(2) for m in matches]
        width = len(indices[0])
        start = indices[0]
        end = indices[-1]
        dir_prefix = os.path.dirname(shards[0])
        return f"s3://{bucket}/{dir_prefix}/{prefix_part}{{{start}..{end}}}{suffix_part}"

    # Fallback: list all URLs explicitly (no brace expansion possible)
    return "\n".join(f"s3://{bucket}/{k}" for k in shards)


def inspect(bucket: str, prefix: str, peek: bool) -> None:
    print(f"Listing shards in s3://{bucket}/{prefix} ...")
    shards = list_shards(bucket, prefix)

    if not shards:
        print("  No .tar files found. Check bucket name and prefix.")
        return

    total_size_mb = 0.0
    client = boto3.client("s3")
    for key in shards:
        resp = client.head_object(Bucket=bucket, Key=key)
        total_size_mb += resp["ContentLength"] / (1024 ** 2)

    print(f"\n  Shards found : {len(shards)}")
    print(f"  Total size   : {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")
    print(f"  First shard  : {shards[0]}")
    print(f"  Last shard   : {shards[-1]}")

    pattern = build_pattern(bucket, prefix, shards)
    print(f"\n  Pattern for configs/train.yaml:")
    print(f'    train_shards: "{pattern}"')

    if peek:
        peek_shard(bucket, shards[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect WebDataset shards in S3")
    parser.add_argument("--s3_bucket", required=True,             help="S3 bucket name")
    parser.add_argument("--s3_prefix", required=True,             help="Prefix (folder) inside the bucket")
    parser.add_argument("--peek",      action="store_true",       help="Stream first shard and show sample keys")
    args = parser.parse_args()

    inspect(args.s3_bucket, args.s3_prefix, args.peek)
