"""Lossless, read-only image IO.

Rules
-----
* Originals are opened read-only and returned in their native dtype (uint8 / uint16 / float32).
  No scaling, no bit shifting, no colour conversion is applied implicitly.
* 10/12-bit sensor data stored in 16-bit containers is returned as-is; the *effective* bit depth is
  metadata (session.bit_depth) used only by explicit normalisation policies.
* Unicode paths on Windows are supported (cv2.imread does not handle them, so we decode from bytes).
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

SUPPORTED_EXT = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".pgm"}


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    dtype: str
    channels: int
    container_bits: int


def read_image(path: str | os.PathLike) -> np.ndarray:
    """Read an image without any value modification. Multichannel images are returned as-is (H, W, C)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(str(p))
        # Some TIFFs come as (1, H, W) pages
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
    else:
        buf = np.fromfile(str(p), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"Cannot decode image: {p}")
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # cv2 gives BGR(A); convert to RGB(A) so channel semantics are standard
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(arr)


def to_mono(arr: np.ndarray) -> np.ndarray:
    """Return a single-channel view. Grey images that were saved as RGB (identical channels) are reduced
    losslessly; true colour images are converted with ITU-R 601 luma in float then cast back to the input
    dtype (this is a lossy operation and is flagged by `is_true_color`)."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    rgb = arr[..., :3]
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return np.ascontiguousarray(rgb[..., 0])
    f = rgb.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        y = np.clip(np.rint(y), info.min, info.max)
    return y.astype(arr.dtype)


def is_true_color(arr: np.ndarray) -> bool:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    rgb = arr[..., :3]
    return not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))


def probe_image(path: str | os.PathLike) -> ImageInfo:
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        with tifffile.TiffFile(str(p)) as tf:
            page = tf.pages[0]
            shape = page.shape
            dtype = page.dtype
            if len(shape) == 3 and shape[0] == 1:
                shape = shape[1:]
            h, w = shape[0], shape[1]
            ch = shape[2] if len(shape) == 3 else 1
            return ImageInfo(int(w), int(h), str(dtype), int(ch), int(np.dtype(dtype).itemsize * 8))
    arr = read_image(p)
    ch = arr.shape[2] if arr.ndim == 3 else 1
    return ImageInfo(arr.shape[1], arr.shape[0], str(arr.dtype), ch, arr.dtype.itemsize * 8)


def estimate_effective_bits(arr: np.ndarray) -> int:
    """Heuristic estimate of the sensor bit depth from the max value (only a hint; user metadata wins)."""
    if arr.dtype == np.uint8:
        return 8
    if not np.issubdtype(arr.dtype, np.integer):
        return 32
    m = int(arr.max()) if arr.size else 0
    for bits in (8, 10, 12, 14, 16):
        if m < (1 << bits):
            return bits
    return arr.dtype.itemsize * 8


def file_sha256(path: str | os.PathLike, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_image_lossless(path: str | os.PathLike, arr: np.ndarray) -> None:
    """Write PNG/TIFF without value changes (used for derived data and synthetic data only)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".tif", ".tiff"):
        tifffile.imwrite(str(p), arr)
        return
    out = arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        out = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if arr.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(p.suffix, out)
    if not ok:
        raise IOError(f"Cannot encode {p}")
    enc.tofile(str(p))


class ImageCache:
    """Thread-safe LRU cache bounded by bytes, so that large datasets are never fully loaded in memory."""

    def __init__(self, max_bytes: int = 1 << 30):
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, path: str) -> np.ndarray:
        key = str(path)
        with self._lock:
            arr = self._d.get(key)
            if arr is not None:
                self._d.move_to_end(key)
                return arr
        arr = to_mono(read_image(key))
        arr.setflags(write=False)  # cached originals are immutable
        with self._lock:
            if key not in self._d:
                self._d[key] = arr
                self._bytes += arr.nbytes
                while self._bytes > self.max_bytes and len(self._d) > 1:
                    _, old = self._d.popitem(last=False)
                    self._bytes -= old.nbytes
        return arr

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


_GLOBAL_CACHE = ImageCache()


def load_mono(path: str | os.PathLike) -> np.ndarray:
    """Read-only, cached, single-channel original image."""
    return _GLOBAL_CACHE.get(str(path))


def global_cache() -> ImageCache:
    return _GLOBAL_CACHE
