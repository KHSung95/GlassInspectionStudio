"""Full-resolution tiling with overlap and exact restoration to original coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)  # last tile aligned to the border: no padding, no resizing
    return s


def make_tiles(height: int, width: int, tile: int, overlap: int = 0) -> list[Tile]:
    """Tiles that cover the whole image at native resolution. Tiles never extend outside the image unless
    the image itself is smaller than the tile (then the tile is the image)."""
    if tile <= 0:
        raise ValueError("tile must be > 0")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    stride = tile - overlap
    tw, th = min(tile, width), min(tile, height)
    return [Tile(x, y, tw, th) for y in _starts(height, tile, stride) for x in _starts(width, tile, stride)]


def extract(img: np.ndarray, t: Tile) -> np.ndarray:
    return img[t.y0:t.y1, t.x0:t.x1]


def pad_to(arr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad a (H, W) tile up to (size, size) for models that need fixed input; returns original hw."""
    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr, (h, w)
    ph, pw = size - h, size - w
    mode = "reflect" if (h > 1 and w > 1 and ph < h and pw < w) else "edge"
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad, mode=mode), (h, w)


class Stitcher:
    """Accumulates per-tile maps back into the original image coordinate system.

    mode='mean' : average over overlaps (with optional cosine window to reduce seams)
    mode='max'  : max over overlaps (conservative for defect scores)
    """

    def __init__(self, height: int, width: int, channels: int = 1, mode: str = "max", window: bool = False):
        self.mode = mode
        self.window = window
        self.shape = (channels, height, width)
        if mode == "max":
            self.acc = np.full(self.shape, -np.inf, dtype=np.float32)
        else:
            self.acc = np.zeros(self.shape, dtype=np.float32)
            self.wsum = np.zeros((height, width), dtype=np.float32)

    @staticmethod
    def _win(h: int, w: int) -> np.ndarray:
        wy = np.hanning(h + 2)[1:-1] if h > 2 else np.ones(h)
        wx = np.hanning(w + 2)[1:-1] if w > 2 else np.ones(w)
        return np.outer(wy, wx).astype(np.float32) + 1e-3

    def add(self, t: Tile, m: np.ndarray) -> None:
        if m.ndim == 2:
            m = m[None]
        m = m[:, : t.h, : t.w]
        if self.mode == "max":
            sl = self.acc[:, t.y0:t.y1, t.x0:t.x1]
            np.maximum(sl, m, out=sl)
        else:
            wgt = self._win(t.h, t.w) if self.window else np.ones((t.h, t.w), np.float32)
            self.acc[:, t.y0:t.y1, t.x0:t.x1] += m * wgt
            self.wsum[t.y0:t.y1, t.x0:t.x1] += wgt

    def result(self) -> np.ndarray:
        if self.mode == "max":
            out = self.acc.copy()
            out[~np.isfinite(out)] = 0.0
            return out
        return self.acc / np.maximum(self.wsum, 1e-12)
