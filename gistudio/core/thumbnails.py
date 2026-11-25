"""On-disk thumbnail cache (keyed by sha256) so browsing large datasets does not load originals."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .image_io import read_image, to_mono, write_image_lossless
from .preprocessing import DisplayTransform


def thumbnail(cache_dir: Path, sha256: str, path: str, size: int = 192, bit_depth: int | None = None) -> np.ndarray:
    p = Path(cache_dir) / f"{sha256[:2]}/{sha256}_{size}.png"
    if p.exists():
        buf = np.fromfile(str(p), dtype=np.uint8)
        t = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            return t
    img = to_mono(read_image(path))
    disp = DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=bit_depth).apply(img)
    h, w = disp.shape
    s = size / max(h, w)
    t = cv2.resize(disp, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    write_image_lossless(p, t)
    return t
