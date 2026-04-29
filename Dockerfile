# Base: official PyTorch image with CUDA 12.1 — matches RunPod A100/H100 drivers
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# awscli: required for the "pipe:aws s3 cp ..." streaming pattern
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Override at runtime: docker run ... python train.py configs/my_experiment.yaml
CMD ["python", "train.py", "configs/train.yaml"]
