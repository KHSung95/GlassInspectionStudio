"""Preprocessing — strictly separated into two independent families.

* ``DisplayTransform``   : visualisation only (window/level, gamma, CLAHE, invert...). Output uint8 for the
                           screen. NEVER used as model input.
* ``InputNormalization`` : the fixed, serialisable policy applied identically during training and inference.
                           It is stored inside every checkpoint / run record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


def dtype_max(arr: np.ndarray, bit_depth: int | None = None) -> float:
    if bit_depth:
        return float((1 << int(bit_depth)) - 1)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    return 1.0


# --------------------------------------------------------------------------------------------- display
@dataclass
class DisplayTransform:
    window: float | None = None   # width of the value window (raw units). None -> auto (full range or stretch)
    level: float | None = None    # centre of the window (raw units)
    gamma: float = 1.0
    invert: bool = False
    stretch_percentiles: tuple[float, float] | None = None  # e.g. (0.5, 99.5): contrast stretch
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    bit_depth: int | None = None

    def value_range(self, arr: np.ndarray) -> tuple[float, float]:
        if self.window is not None and self.level is not None:
            lo = self.level - self.window / 2.0
            hi = self.level + self.window / 2.0
        elif self.stretch_percentiles is not None:
            sub = arr[:: max(1, arr.shape[0] // 512), :: max(1, arr.shape[1] // 512)]
            lo, hi = np.percentile(sub, self.stretch_percentiles)
        else:
            lo, hi = 0.0, dtype_max(arr, self.bit_depth)
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Return a NEW uint8 array for display. Input is never modified."""
        lo, hi = self.value_range(arr)
        if arr.dtype == np.uint8 and lo == 0 and hi == 255 and self.gamma == 1.0:
            out = arr.copy()
        elif np.issubdtype(arr.dtype, np.integer) and arr.dtype.itemsize <= 2:
            # LUT path: fast for 8/16 bit
            n = 256 if arr.dtype == np.uint8 else 65536
            x = np.arange(n, dtype=np.float32)
            y = np.clip((x - lo) / (hi - lo), 0, 1)
            if self.gamma != 1.0:
                y = np.power(y, 1.0 / self.gamma)
            lut = (y * 255.0 + 0.5).astype(np.uint8)
            out = lut[arr]
        else:
            y = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1)
            if self.gamma != 1.0:
                y = np.power(y, 1.0 / self.gamma)
            out = (y * 255.0 + 0.5).astype(np.uint8)
        if self.clahe:
            c = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
            out = c.apply(out)
        if self.invert:
            out = 255 - out
        return out

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------- model input
@dataclass
class InputNormalization:
    """Deterministic intensity normalisation for models.

    mode='fixed_range': x / (2**bit_depth - 1)  — preserves absolute brightness & low-contrast defects.
    No per-image contrast normalisation by default, because it can erase transparency/intensity defects.
    """

    mode: str = "fixed_range"         # fixed_range | fixed_affine
    bit_depth: int | None = None      # None -> use container max
    offset: float = 0.0               # fixed_affine: (x - offset) / scale
    scale: float = 1.0
    clip: bool = True

    def apply(self, arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float32)
        if self.mode == "fixed_range":
            x = x / dtype_max(arr, self.bit_depth)
        elif self.mode == "fixed_affine":
            x = (x - self.offset) / self.scale
        else:
            raise ValueError(f"Unknown normalization mode {self.mode}")
        if self.clip and self.mode == "fixed_range":
            x = np.clip(x, 0.0, 1.0)
        return x

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InputNormalization":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def histogram(arr: np.ndarray, bins: int = 256, bit_depth: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    hi = dtype_max(arr, bit_depth)
    sub = arr[:: max(1, arr.shape[0] // 1024), :: max(1, arr.shape[1] // 1024)]
    h, e = np.histogram(sub, bins=bins, range=(0, hi + 1))
    return h, e


def roi_stats(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> dict:
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return {}
    r = arr[y0:y1, x0:x1].astype(np.float64)
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1, "n": int(r.size),
        "mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()), "max": float(r.max()),
        "median": float(np.median(r)),
    }


@dataclass
class FeaturePreprocess:
    """Optional, explicit preprocessing steps for classical algorithms (recorded in run history)."""

    gaussian_sigma: float = 0.0
    median_ksize: int = 0
    extra: dict = field(default_factory=dict)

    def apply(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32)
        if self.median_ksize and self.median_ksize >= 3:
            k = int(self.median_ksize) | 1
            x = cv2.medianBlur(x, k) if k <= 5 else cv2.medianBlur(x.astype(np.uint16), k).astype(np.float32)
        if self.gaussian_sigma > 0:
            x = cv2.GaussianBlur(x, (0, 0), self.gaussian_sigma)
        return x

    def to_dict(self) -> dict:
        return asdict(self)
