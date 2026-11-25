"""Ridge detector finds a thin curved dark void and ignores a compact dot."""
import cv2
import numpy as np

from gistudio.algorithms.base import create


def test_ridge_void_detects_thin_curve_not_dot():
    rng = np.random.default_rng(0)
    img = (180 + rng.normal(0, 3, (200, 300))).astype(np.float32)
    cv2.ellipse(img, (150, 120), (80, 40), 0, 200, 330, 120, 2)      # boomerang-like thin dark arc
    cv2.circle(img, (40, 40), 3, 100, -1)                             # compact dark dot (black spot)
    img = np.clip(img, 0, 255).astype(np.uint8)
    res = create("classical.ridge_void", threshold=6.0, polarity="dark").run(img)
    assert res.candidates, "arc not detected"
    top = res.candidates[0]
    g = top.geometry
    assert g["x1"] - g["x0"] > 60 and top.features["elongation"] >= 3
    assert not any(c.geometry["x0"] <= 40 <= c.geometry["x1"] and c.geometry["y0"] <= 40 <= c.geometry["y1"]
                   for c in res.candidates), "compact dot must not be reported as a linear void"
