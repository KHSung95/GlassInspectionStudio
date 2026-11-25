"""Low-level classical image-processing operators (float32, original resolution)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False) if img.dtype == np.float32 else img.astype(np.float32)


def noise_sigma(img: np.ndarray) -> float:
    """Robust (MAD) estimate of pixel noise from horizontal first differences. Structures (edges, digits) are
    sparse and mostly rejected by the median."""
    x = f32(img)
    step = max(1, x.shape[0] // 512)
    d = np.diff(x[::step], axis=1).ravel()
    mad = np.median(np.abs(d - np.median(d)))
    return float(max(1.4826 * mad / np.sqrt(2.0), 1e-3 * (float(x.max()) - float(x.min()) + 1.0), 1e-6))


def noise_sigma_meta(img: np.ndarray, meta: dict | None) -> float:
    """Use a frame-level noise sigma when the caller provides one (ROI / crop runs), so SNR scores and
    thresholds stay identical to full-frame runs."""
    if meta and meta.get("noise_sigma"):
        return float(meta["noise_sigma"])
    return noise_sigma(img)


def gauss(x: np.ndarray, s: float) -> np.ndarray:
    return x if s <= 0 else cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)


def blackhat(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(x, cv2.MORPH_BLACKHAT, se, borderType=cv2.BORDER_REFLECT)


def tophat(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(x, cv2.MORPH_TOPHAT, se, borderType=cv2.BORDER_REFLECT)


def dog_dark(x: np.ndarray, s1: float, s2: float) -> np.ndarray:
    """Positive response for structures darker than their surroundings at scale ~s1."""
    return gauss(x, s2) - gauss(x, s1)


def log_dark_multiscale(x: np.ndarray, sigmas: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Scale-normalised LoG (sigma^2 * Laplacian) max over scales; positive for dark blobs.
    Returns (response, argmax-scale sigma map)."""
    best = np.full(x.shape, -np.inf, np.float32)
    best_s = np.zeros(x.shape, np.float32)
    for s in sigmas:
        g = gauss(x, s)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT)
        r = (s * s) * lap   # dark blob -> positive laplacian at centre
        m = r > best
        best[m] = r[m]
        best_s[m] = s
    return best, best_s


def background_median(x: np.ndarray, k: int, downsample: int = 4) -> np.ndarray:
    """Large-kernel median background via downsample -> median -> upsample (fast for 16-bit data)."""
    h, w = x.shape
    ds = max(1, int(downsample))
    small = cv2.resize(x, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
    kk = max(3, (int(k) // ds) | 1)
    bg = ndi.median_filter(small, size=kk, mode="reflect")
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def local_mean_std(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(3, int(k) | 1)
    m = cv2.boxFilter(x, cv2.CV_32F, (k, k), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(x * x, cv2.CV_32F, (k, k), borderType=cv2.BORDER_REFLECT)
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def gradient_mag(x: np.ndarray, s: float = 1.0) -> np.ndarray:
    g = gauss(x, s)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def local_contrast_at(img: np.ndarray, x: float, y: float, r_in: float = 2.0, r_out: float = 8.0) -> dict:
    """Disc vs annulus statistics around (x, y) — used for candidate features & angular analysis."""
    h, w = img.shape
    R = int(np.ceil(r_out)) + 1
    cx, cy = x - 0.5, y - 0.5
    x0, x1 = max(0, int(cx) - R), min(w, int(cx) + R + 1)
    y0, y1 = max(0, int(cy) - R), min(h, int(cy) + R + 1)
    if x1 <= x0 or y1 <= y0:
        return {"inner": 0.0, "outer": 0.0, "contrast": 0.0, "inner_min": 0.0}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    patch = img[y0:y1, x0:x1].astype(np.float32)
    inner = patch[d2 <= r_in * r_in]
    outer = patch[(d2 > (r_in + 1) ** 2) & (d2 <= r_out * r_out)]
    if inner.size == 0 or outer.size == 0:
        return {"inner": 0.0, "outer": 0.0, "contrast": 0.0, "inner_min": 0.0}
    om = float(np.median(outer))
    return {"inner": float(inner.mean()), "outer": om, "contrast": om - float(inner.mean()),
            "inner_min": float(inner.min()), "outer_std": float(outer.std())}


def components(score: np.ndarray, thr: float, min_area: int = 1, max_area: int = 10**9,
               max_count: int = 2000, valid: np.ndarray | None = None) -> list[dict]:
    """Connected components of score >= thr with peak position, area, bbox, peak score."""
    mask = score >= thr
    if valid is not None:
        mask &= valid
    if not mask.any():
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    idx = np.arange(1, n)
    maxpos = ndi.maximum_position(score, lab, idx)
    maxval = ndi.maximum(score, lab, idx)
    out = []
    for i, (py, px), v in zip(idx, maxpos, maxval):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                    cv2.CC_STAT_HEIGHT))
        out.append({"label": int(i), "peak_x": px + 0.5, "peak_y": py + 0.5, "score": float(v), "area": area,
                    "bbox": (float(x), float(y), float(x + w), float(y + h))})
    out.sort(key=lambda d: -d["score"])
    return out[:max_count]
