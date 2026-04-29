import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_probs).gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt) ** self.gamma * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        class_weights=None,
        gamma=1.0,
        dice_weight=1.0,
        focal_weight=1.0,
        ignore_index=0,
        eps=1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        device = logits.device

        valid_mask = targets != self.ignore_index if self.ignore_index >= 0 else torch.ones_like(targets, dtype=torch.bool)

        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets_clamped, num_classes=C).permute(0, 3, 1, 2).float()

        pt = (probs * targets_onehot).sum(1)
        log_pt = (log_probs * targets_onehot).sum(1)

        alpha = self.class_weights.to(device)
        alpha_factor = (targets_onehot * alpha.view(1, C, 1, 1)).sum(1)

        focal_per_pixel = -alpha_factor * ((1 - pt) ** self.gamma) * log_pt
        focal_loss = (focal_per_pixel * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        valid_mask_c = valid_mask.unsqueeze(1).float()
        probs_masked = probs * valid_mask_c
        tgt_masked = targets_onehot * valid_mask_c

        start_c = 1 if self.ignore_index == 0 else 0
        probs_flat = probs_masked[:, start_c:].reshape(B, C - start_c, -1)
        tgt_flat = tgt_masked[:, start_c:].reshape(B, C - start_c, -1)

        intersection = (probs_flat * tgt_flat).sum(-1)
        union = probs_flat.sum(-1) + tgt_flat.sum(-1)
        dice_loss = (1 - (2 * intersection + self.eps) / (union + self.eps)).mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
