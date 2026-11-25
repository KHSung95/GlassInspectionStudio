"""Supervised models: high-resolution U-Net and multi-scale Patch CNN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    g = max(d for d in range(1, min(8, c) + 1) if c % d == 0)
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False), _norm(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False), _norm(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContextModule(nn.Module):
    """Multi-scale context at the bottleneck (parallel dilated convs, ASPP-lite)."""

    def __init__(self, c: int, rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=r, dilation=r, bias=False),
                                                     _norm(c), nn.SiLU(inplace=True)) for r in rates])
        self.fuse = nn.Sequential(nn.Conv2d(c * len(rates), c, 1, bias=False), _norm(c), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], 1)) + x


class HRUNet(nn.Module):
    """Lightweight U-Net for tiny-defect segmentation / heatmaps.

    * First stage runs at FULL input resolution (no strided stem) so 1-3px structures keep their signal.
    * Skip connections at every level; output at the input resolution.
    * Optional multi-scale context module at the bottleneck.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 3, base: int = 16, depth: int = 4, context: bool = True):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, out_ch=out_ch, base=base, depth=depth, context=context)
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.stem = ConvBlock(in_ch, chs[0])
        self.down = nn.ModuleList([ConvBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.ctx = ContextModule(chs[-1]) if context else nn.Identity()
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([ConvBlock(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        nn.init.constant_(self.head.bias, -2.19)  # prior p~0.1: defects are rare (CenterNet/RetinaNet practice)
        self.divisor = 2 ** depth

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (self.divisor - h % self.divisor) % self.divisor
        pw = (self.divisor - w % self.divisor) % self.divisor
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        skips = [self.stem(x)]
        y = skips[0]
        for d in self.down:
            y = d(F.max_pool2d(y, 2))
            skips.append(y)
        y = self.ctx(skips.pop())
        for up, dec in zip(self.up, self.dec):
            y = up(y)
            y = dec(torch.cat([y, skips.pop()], 1))
        out = self.head(y)
        return out[..., :h, :w]


class PatchBranch(nn.Module):
    def __init__(self, c: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), _norm(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), _norm(c), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), _norm(2 * c), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1), _norm(4 * c), nn.SiLU(), nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class MultiScalePatchCNN(nn.Module):
    """Classifies a candidate location from several context crops (e.g. 32, 128, 256 px).

    Each crop is resampled to `branch_px` (small crops are kept at native resolution when crop <= branch_px).
    """

    def __init__(self, n_classes: int, crop_sizes=(32, 128, 256), branch_px: int = 64, width: int = 32):
        super().__init__()
        self.cfg = dict(n_classes=n_classes, crop_sizes=list(crop_sizes), branch_px=branch_px, width=width)
        self.branches = nn.ModuleList([PatchBranch(width) for _ in crop_sizes])
        self.head = nn.Sequential(nn.Linear(4 * width * len(crop_sizes), 128), nn.SiLU(), nn.Dropout(0.2),
                                  nn.Linear(128, n_classes))

    def forward(self, crops: list[torch.Tensor]):
        return self.head(torch.cat([b(c) for b, c in zip(self.branches, crops)], 1))
