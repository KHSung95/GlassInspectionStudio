"""Geometry helpers: coordinate conventions, view transforms, circular angle math, shape rasterisation.

Coordinate convention (used everywhere, stored in DB):
  * Image pixel (col=i, row=j) covers the continuous square [i, i+1) x [j, j+1).
  * Pixel centre is (i + 0.5, j + 0.5).
  * Geometry is stored in continuous original-image coordinates (float, x right, y down), never in
    display/zoomed coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- view transform
@dataclass
class ViewTransform:
    """view = scale * image + offset (uniform scale, no rotation)."""

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def image_to_view(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def view_to_image(self, vx: float, vy: float) -> tuple[float, float]:
        return (vx - self.offset_x) / self.scale, (vy - self.offset_y) / self.scale

    def zoom_about(self, factor: float, vx: float, vy: float) -> None:
        """Zoom keeping the image point under view position (vx, vy) fixed."""
        ix, iy = self.view_to_image(vx, vy)
        self.scale *= factor
        self.offset_x = vx - ix * self.scale
        self.offset_y = vy - iy * self.scale

    def pan(self, dvx: float, dvy: float) -> None:
        self.offset_x += dvx
        self.offset_y += dvy


def pixel_index(x: float, y: float) -> tuple[int, int]:
    """Continuous image coordinate -> integer pixel index (col, row)."""
    return int(math.floor(x)), int(math.floor(y))


# --------------------------------------------------------------------------- circular angles
def wrap_deg(a: float) -> float:
    """Wrap to [0, 360)."""
    r = math.fmod(a, 360.0)
    if r < 0:
        r += 360.0
    if r >= 360.0:  # guards fmod rounding (e.g. -1e-15 + 360)
        r -= 360.0
    return r


def circ_diff(a: float, b: float) -> float:
    """Signed smallest difference a - b in (-180, 180]."""
    d = wrap_deg(a - b)
    return d - 360.0 if d > 180.0 else d


def circ_dist(a: float, b: float) -> float:
    return abs(circ_diff(a, b))


def nearest_angle_index(angles: list[float] | np.ndarray, target: float) -> int:
    a = np.asarray(angles, dtype=np.float64)
    d = np.abs((a - target + 180.0) % 360.0 - 180.0)
    return int(np.argmin(d))


def circular_intervals(angles_sorted: list[float], visible: list[bool], step: float) -> list[tuple[float, float]]:
    """Contiguous visible arcs on the circle. Merges an arc that crosses 0/360.
    Returns (start, end) with end possibly > 360 when crossing (end = start + length)."""
    n = len(angles_sorted)
    if n == 0 or not any(visible):
        return []
    if all(visible):
        return [(0.0, 360.0)]
    # rotate so that index 0 is a non-visible sample
    k0 = visible.index(False)
    order = list(range(k0, n)) + list(range(0, k0))
    arcs, cur = [], None
    for idx in order:
        if visible[idx]:
            a = angles_sorted[idx]
            if cur is None:
                cur = [a, a]
            else:
                cur[1] = a
        elif cur is not None:
            arcs.append(cur)
            cur = None
    if cur is not None:
        arcs.append(cur)
    out = []
    for s, e in arcs:
        length = wrap_deg(e - s) + step
        out.append((s - step / 2.0, s - step / 2.0 + length))
    return out


# --------------------------------------------------------------------------- shapes
def geometry_bbox(gtype: str, g: dict) -> tuple[float, float, float, float]:
    if gtype == "point":
        r = float(g.get("r", 1.0))
        return g["x"] - r, g["y"] - r, g["x"] + r, g["y"] + r
    if gtype == "bbox":
        return g["x0"], g["y0"], g["x1"], g["y1"]
    if gtype == "polygon":
        pts = np.asarray(g["points"], dtype=np.float64)
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())
    if gtype == "mask":
        return g["x0"], g["y0"], g["x0"] + g["w"], g["y0"] + g["h"]
    raise ValueError(gtype)


def geometry_center(gtype: str, g: dict) -> tuple[float, float]:
    if gtype == "point":
        return float(g["x"]), float(g["y"])
    if gtype == "polygon":
        pts = np.asarray(g["points"], dtype=np.float64)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x0, y0, x1, y1 = geometry_bbox(gtype, g)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def rasterize(gtype: str, g: dict, shape: tuple[int, int], mask_array: np.ndarray | None = None,
              point_radius: float | None = None) -> np.ndarray:
    """Rasterise a geometry into a boolean mask of image `shape` (H, W). A pixel is inside when its
    centre is inside the shape."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    if gtype == "point":
        r = point_radius if point_radius is not None else float(g.get("r", 1.0))
        cx, cy = g["x"] - 0.5, g["y"] - 0.5  # to pixel-centre index space
        x0, x1 = max(0, int(math.floor(cx - r))), min(w - 1, int(math.ceil(cx + r)))
        y0, y1 = max(0, int(math.floor(cy - r))), min(h - 1, int(math.ceil(cy + r)))
        if x1 >= x0 and y1 >= y0:
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            m[y0:y1 + 1, x0:x1 + 1] = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r + 1e-9)
    elif gtype == "bbox":
        x0, x1 = sorted((g["x0"], g["x1"]))
        y0, y1 = sorted((g["y0"], g["y1"]))
        # pixel i inside iff x0 <= i+0.5 < x1
        i0, i1 = max(0, int(math.ceil(x0 - 0.5))), min(w, int(math.ceil(x1 - 0.5)))
        j0, j1 = max(0, int(math.ceil(y0 - 0.5))), min(h, int(math.ceil(y1 - 0.5)))
        if i1 > i0 and j1 > j0:
            m[j0:j1, i0:i1] = 1
    elif gtype == "polygon":
        pts = np.asarray(g["points"], dtype=np.float64) - 0.5
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    elif gtype == "mask":
        if mask_array is None:
            raise ValueError("mask geometry requires mask_array")
        x0, y0 = int(g["x0"]), int(g["y0"])
        mh, mw = mask_array.shape
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(w, x0 + mw), min(h, y0 + mh)
        if xs1 > xs0 and ys1 > ys0:
            m[ys0:ys1, xs0:xs1] = (mask_array[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] > 0)
    else:
        raise ValueError(gtype)
    return m.astype(bool)


def gaussian_heatmap(shape: tuple[int, int], points: list[tuple[float, float]], sigma: float,
                     truncate: float = 3.0) -> np.ndarray:
    """Point annotations -> training heatmap (peak 1.0 at each point, max-combined).

    NOTE: the heatmap footprint is a *training target* shape controlled by `sigma`; it is NOT the physical
    size of the defect.
    """
    h, w = shape
    hm = np.zeros((h, w), dtype=np.float32)
    rad = max(1, int(math.ceil(truncate * sigma)))
    for (x, y) in points:
        cx, cy = x - 0.5, y - 0.5
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = max(0, ix - rad), min(w - 1, ix + rad)
        y0, y1 = max(0, iy - rad), min(h - 1, iy + rad)
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
        hm[y0:y1 + 1, x0:x1 + 1] = np.maximum(hm[y0:y1 + 1, x0:x1 + 1], g)
        # the pixel containing the point is the exact peak (1.0) so peak-based losses always see a positive
        px, py = int(math.floor(x)), int(math.floor(y))
        if 0 <= px < w and 0 <= py < h:
            hm[py, px] = 1.0
    return hm
