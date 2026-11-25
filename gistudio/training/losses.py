"""Masked losses for partial labels.

Every loss takes (logits, target, valid) with shapes (B, C, H, W); `valid` is a bool mask. Pixels with
valid=False (Unknown / Ignore) contribute NOTHING to the loss or its gradient. Batches without any valid
pixel return an exact zero that is still connected to the graph (so `backward()` is safe).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -1


def _zero(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def masked_bce(logits, target, valid, pos_weight: float | None = None) -> torch.Tensor:
    v = valid.bool()
    if not v.any():
        return _zero(logits)
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    lm = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    return lm[v].mean()


def masked_focal(logits, target, valid, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    v = valid.bool()
    if not v.any():
        return _zero(logits)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    lm = a * (1 - pt) ** gamma * ce
    return lm[v].sum() / v.sum().clamp_min(1)


def masked_dice(logits, target, valid, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice computed only on valid pixels, per channel, averaged over channels with valid pixels."""
    v = valid.float()
    if v.sum() == 0:
        return _zero(logits)
    p = torch.sigmoid(logits) * v
    t = target * v
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    has = v.sum(dims) > 0
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return dice[has].mean() if has.any() else _zero(logits)


def masked_heatmap_focal(logits, target, valid, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style penalty-reduced focal loss for Gaussian point heatmaps (masked).
    Positives are pixels where target == 1 (rounded peak, >= 0.99)."""
    v = valid.bool()
    if not v.any():
        return _zero(logits)
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.99) & v
    neg = (~(target >= 0.99)) & v
    pos_loss = -(torch.log(p) * (1 - p) ** alpha)[pos].sum()
    neg_loss = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta)[neg].sum()
    n_pos = pos.sum()
    return (pos_loss + neg_loss) / n_pos.clamp_min(1)


def masked_mse(logits, target, valid) -> torch.Tensor:
    v = valid.bool()
    if not v.any():
        return _zero(logits)
    return ((torch.sigmoid(logits) - target) ** 2)[v].mean()


LOSSES = {
    "bce": masked_bce,
    "focal": masked_focal,
    "dice": masked_dice,
    "heatmap_focal": masked_heatmap_focal,
    "mse": masked_mse,
}


def combined_loss(logits, target, valid, spec: dict[str, str], channels: list[str],
                  pos_weight: float | None = None) -> tuple[torch.Tensor, dict]:
    """spec: channel -> '+'-joined loss names, e.g. {'black_spot': 'heatmap_focal', 'bubble': 'bce+dice'}."""
    total = _zero(logits)
    parts = {}
    for ci, ch in enumerate(channels):
        names = spec.get(ch, "bce+dice").split("+")
        sl = slice(ci, ci + 1)
        for n in names:
            n = n.strip()
            if n == "bce":
                l = masked_bce(logits[:, sl], target[:, sl], valid[:, sl], pos_weight)
            else:
                l = LOSSES[n](logits[:, sl], target[:, sl], valid[:, sl])
            parts[f"{ch}/{n}"] = float(l.detach())
            total = total + l
        parts[f"{ch}/valid_px"] = int(valid[:, sl].sum())
    return total, parts
