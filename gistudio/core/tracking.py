"""Optional tracking aids (template matching). Results are *suggestions* only (confirmed=0)."""
from __future__ import annotations

import cv2
import numpy as np


def track_point_template(img_a: np.ndarray, img_b: np.ndarray, x: float, y: float, patch: int = 24,
                         search: int = 64) -> tuple[float, float, float]:
    """Find the location in img_b of the patch around (x, y) in img_a. Returns (x, y, ncc_score).
    Score < ~0.5 should be treated as 'Not visible / uncertain' by the caller."""
    h, w = img_a.shape
    cx, cy = int(x), int(y)
    half = patch // 2
    ax0, ay0 = max(0, cx - half), max(0, cy - half)
    ax1, ay1 = min(w, cx + half), min(h, cy + half)
    tpl = img_a[ay0:ay1, ax0:ax1].astype(np.float32)
    sx0, sy0 = max(0, cx - search), max(0, cy - search)
    sx1, sy1 = min(w, cx + search), min(h, cy + search)
    win = img_b[sy0:sy1, sx0:sx1].astype(np.float32)
    if tpl.size == 0 or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1] or tpl.std() == 0:
        return x, y, 0.0
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    nx = sx0 + loc[0] + (x - ax0)
    ny = sy0 + loc[1] + (y - ay0)
    return float(nx), float(ny), float(mx)
