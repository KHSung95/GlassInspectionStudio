"""Physically conservative augmentations. They never resize, never apply per-image contrast normalisation
and keep blur below the optical PSF range configured here. Synthetic defects are flagged and are for
training support only — evaluation must use real defects."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class AugConfig:
    gain_range: tuple[float, float] = (0.92, 1.08)     # multiplicative brightness (gain/exposure variation)
    offset_range: tuple[float, float] = (-0.02, 0.02)  # additive (fraction of full scale)
    noise_std_range: tuple[float, float] = (0.0, 0.004)
    hflip: bool = True        # mirror along the rotation axis direction (configurable: may be invalid for text)
    vflip: bool = False       # bottles are not upside-down: off by default
    blur_prob: float = 0.15
    blur_sigma_max: float = 0.6
    synth_spots_prob: float = 0.3
    synth_spots_max: int = 3
    synth_radius_range: tuple[float, float] = (0.5, 2.0)
    synth_contrast_range: tuple[float, float] = (0.04, 0.35)  # fractional darkening at centre
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AugConfig":
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


def _gauss_blur(x: np.ndarray, s: float) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(x, (0, 0), s)


def augment(x: np.ndarray, target: np.ndarray, valid: np.ndarray, cfg: AugConfig, rng: np.random.Generator,
            spot_channel: int | None, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """x: (H, W) float in [0,1]; target/valid: (C, H, W). Returns augmented copies + number of synthetic spots."""
    if not cfg.enabled:
        return x, target, valid, 0
    x = x.copy(); target = target.copy(); valid = valid.copy()
    n_syn = 0
    if cfg.hflip and rng.random() < 0.5:
        x, target, valid = x[:, ::-1], target[:, :, ::-1], valid[:, :, ::-1]
    if cfg.vflip and rng.random() < 0.5:
        x, target, valid = x[::-1], target[:, ::-1], valid[:, ::-1]
    x = np.ascontiguousarray(x); target = np.ascontiguousarray(target); valid = np.ascontiguousarray(valid)
    # synthetic black spots only on pixels known to be negative for the spot channel (never on unknown areas,
    # where a real defect might already be)
    if spot_channel is not None and rng.random() < cfg.synth_spots_prob:
        free = valid[spot_channel] & (target[spot_channel] < 0.01)
        ys, xs = np.nonzero(free)
        h, w = x.shape
        for _ in range(int(rng.integers(1, cfg.synth_spots_max + 1))):
            if len(ys) == 0:
                break
            k = int(rng.integers(len(ys)))
            cy, cx = ys[k] + 0.5, xs[k] + 0.5
            r = rng.uniform(*cfg.synth_radius_range)
            c = rng.uniform(*cfg.synth_contrast_range)
            R = int(np.ceil(3 * max(r, sigma))) + 1
            y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
            x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            prof = np.exp(-d2 / (2 * r * r))
            x[y0:y1, x0:x1] *= (1 - c * prof)
            hm = np.exp(-d2 / (2 * sigma * sigma)).astype(np.float32)
            target[spot_channel, y0:y1, x0:x1] = np.maximum(target[spot_channel, y0:y1, x0:x1], hm)
            valid[spot_channel, y0:y1, x0:x1] = True
            n_syn += 1
    if rng.random() < cfg.blur_prob and cfg.blur_sigma_max > 0:
        x = _gauss_blur(x, rng.uniform(0.3, cfg.blur_sigma_max))
    x = x * rng.uniform(*cfg.gain_range) + rng.uniform(*cfg.offset_range)
    ns = rng.uniform(*cfg.noise_std_range)
    if ns > 0:
        x = x + rng.normal(0, ns, x.shape).astype(np.float32)
    return x.astype(np.float32), target, valid, n_syn
