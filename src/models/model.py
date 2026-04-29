import timm
import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build any classification backbone from timm.

    Examples in configs/train.yaml:
        resnet50, resnet101
        vit_base_patch16_224, vit_large_patch16_224
        convnext_base, convnext_large
        efficientnet_b4, efficientnet_b7
        swin_base_patch4_window7_224

    For semantic segmentation:
        Replace with segmentation_models_pytorch:
            import segmentation_models_pytorch as smp
            return smp.Unet(encoder_name="resnet50", classes=cfg.model.num_classes)

    For HuggingFace models (VLM, OCR, multimodal):
        from transformers import AutoModel
        return AutoModel.from_pretrained(cfg.model.name)
    """
    return timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )
