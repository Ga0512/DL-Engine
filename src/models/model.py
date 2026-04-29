import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    task = getattr(cfg, "task", "classification")
    if task == "segmentation":
        return _build_seg_model(cfg)
    return _build_cls_model(cfg)


def _build_cls_model(cfg: DictConfig) -> nn.Module:
    return timm.create_model(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
    )


def _build_seg_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name.lower()
    in_ch = cfg.model.get("in_channels", 3)
    num_classes = cfg.model.num_classes

    if name in ("attention_resunet", "resunet"):
        backbone = cfg.model.get("encoder", "resnet34")
        return AttentionResUNet(in_channels=in_ch, num_classes=num_classes, model_size=backbone)

    # Fallback: segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp
        encoder = cfg.model.get("encoder", "resnet50")
        weights = "imagenet" if cfg.model.pretrained else None
        return getattr(smp, cfg.model.name)(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_ch,
            classes=num_classes,
        )
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Unknown segmentation model '{cfg.model.name}'. "
            f"Use 'attention_resunet' (built-in) or any smp model with `pip install segmentation-models-pytorch`. "
            f"Original error: {e}"
        )


# ---------------------------------------------------------------------------
# AttentionResUNet — ported from github.com/Ga0512/SatSegmentation
# ---------------------------------------------------------------------------

_ENCODER_CH = {
    "resnet34": (64,  64,  128,  256,   512),
    "resnet50": (64, 256,  512, 1024,  2048),
}


class _AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=False)
        self.spatial = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(F_int, 1, 1, bias=False), nn.Sigmoid()
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_l, max(F_l // 16, 1), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(F_l // 16, 1), F_l, 1), nn.Sigmoid(),
        )

    def forward(self, g, x):
        psi = self.spatial(self.W_g(g) + self.W_x(x))
        x_s = x * psi
        return x_s * self.channel(x_s) + x_s


class AttentionResUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, model_size: str = "resnet34"):
        super().__init__()
        assert model_size in _ENCODER_CH, f"model_size must be one of {list(_ENCODER_CH)}"
        ci, e1, e2, e3, e4 = _ENCODER_CH[model_size]

        enc = models.resnet34(weights=None) if model_size == "resnet34" else models.resnet50(weights=None)
        old = enc.conv1
        enc.conv1 = nn.Conv2d(in_channels, old.out_channels, 7, stride=2, padding=3, bias=False)

        self.initial = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc1 = enc.layer1
        self.enc2 = enc.layer2
        self.enc3 = enc.layer3
        self.enc4 = enc.layer4

        self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att4 = _AttentionGate(e4, e3, e3 // 2)
        self.dec4 = self._block(e4 + e3, e3)

        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att3 = _AttentionGate(e3, e2, e2 // 2)
        self.dec3 = self._block(e3 + e2, e2)

        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att2 = _AttentionGate(e2, e1, e1 // 2)
        self.dec2 = self._block(e2 + e1, e1)

        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = self._block(e1 + ci, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d4 = self.dec4(torch.cat([self.up4(x4), self.att4(self.up4(x4), x3)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(self.up3(d4), x2)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(self.up2(d3), x1)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], 1))

        out = self.final(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out
