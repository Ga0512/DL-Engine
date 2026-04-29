"""
RunPod Serverless Handler — OPTIONAL, only for inference endpoints.

---
IMPORTANT: você provavelmente NÃO precisa deste arquivo para treinamento.

RunPod tem dois modos:

1. Pod (o que você quer para treino)
   - Você aluga uma GPU por hora, faz SSH, e roda: python train.py
   - Sem handler. Sem serverless. Só código Python normal.
   - O pod fica rodando até você parar manualmente ou o job terminar.

2. Serverless (este arquivo)
   - Para APIs de inferência que escalam de 0 → N sob demanda.
   - Você envia uma requisição, um worker fria e processa, depois desliga.
   - Cobre o caso: "quero um endpoint de predict que não fique ligado 24/7".

Se seu objetivo é só treinar, pode deletar este arquivo.
Se futuramente quiser servir o modelo treinado como API no RunPod, use como base.
---

Requisito: pip install runpod
Documentação: https://docs.runpod.io/serverless/workers/handlers/overview
"""

import os
import torch
import runpod
from PIL import Image
import base64
import io

from omegaconf import OmegaConf
from src.models.model import build_model
from src.systems.lightning_module import ClassificationSystem


_model = None
_cfg = None


def _load_model():
    global _model, _cfg
    if _model is not None:
        return _model

    cfg_path = os.environ.get("CONFIG_PATH", "configs/train.yaml")
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/last.ckpt")

    _cfg = OmegaConf.load(cfg_path)
    backbone = build_model(_cfg)
    system = ClassificationSystem.load_from_checkpoint(ckpt_path, model=backbone, cfg=_cfg)
    system.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = system.to(device)
    return _model


def handler(job: dict) -> dict:
    """
    Input payload:
        {
            "input": {
                "image_b64": "<base64-encoded JPEG>"
            }
        }

    Output:
        {
            "class_id": 3,
            "confidence": 0.97
        }
    """
    job_input = job.get("input", {})
    image_b64 = job_input.get("image_b64")

    if not image_b64:
        return {"error": "Missing 'image_b64' in input"}

    try:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = _load_model()
    device = next(model.parameters()).device

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, class_id = probs.max(dim=1)

    return {
        "class_id": class_id.item(),
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
