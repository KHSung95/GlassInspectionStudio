"""Montage of the top-K candidates of a run (native-resolution crops) for visual review / reports.

    python tools/candidate_montage.py <project> <run_id> <out.png> [--frames 13,4] [--k 10] [--crop 96]
Frames are given as positions in path order (0-based).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.core.geometry import geometry_center  # noqa: E402
from gistudio.core.image_io import load_mono  # noqa: E402
from gistudio.core.project import Project  # noqa: E402
from gistudio.inference.engine import candidates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("run", type=int); ap.add_argument("out")
ap.add_argument("--frames", default="0"); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--crop", type=int, default=96); ap.add_argument("--tile", type=int, default=160)
a = ap.parse_args()
p = Project(a.project)
fr = sorted(p.all_frames(), key=lambda f: f["path"])
rows = []
for fi in [int(x) for x in a.frames.split(",")]:
    img = load_mono(fr[fi]["path"]); h, w = img.shape
    tiles = []
    for k, c in enumerate(candidates(p, a.run, fr[fi]["id"])[: a.k]):
        x, y = (int(v) for v in geometry_center(c["geometry_type"], c["geometry"]))
        s = a.crop
        x0, y0 = max(0, min(w - s, x - s // 2)), max(0, min(h - s, y - s // 2))
        t = cv2.resize(img[y0:y0 + s, x0:x0 + s], (a.tile, a.tile), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        label = "f%d #%d %.0f" % (fi, k + 1, c["score"])
        cv2.putText(t, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) < a.k:
        tiles.append(np.zeros((a.tile, a.tile, 3), np.uint8))
    rows.append(np.hstack(tiles))
ok, enc = cv2.imencode(".png", np.vstack(rows)); enc.tofile(a.out)
print(a.out)
