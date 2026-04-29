import os
import boto3
from pathlib import Path


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),  # For R2/MinIO/etc.
    )


def upload_file(local_path: str, bucket: str, s3_key: str) -> None:
    client = get_s3_client()
    client.upload_file(local_path, bucket, s3_key)
    print(f"  uploaded → s3://{bucket}/{s3_key}")


def download_file(bucket: str, s3_key: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()
    client.download_file(bucket, s3_key, local_path)
    print(f"  downloaded s3://{bucket}/{s3_key} → {local_path}")


def list_shards(bucket: str, prefix: str) -> list[str]:
    """List all .tar shards under a prefix, sorted."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    shards = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar"):
                shards.append(f"s3://{bucket}/{obj['Key']}")
    return sorted(shards)
