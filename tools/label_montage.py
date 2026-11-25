"""Native-resolution crops of every confirmed positive label, grouped by class, with measured appearance.

    python tools/label_montage.py <project> <out_dir>
Writes labels_<class>.png montages and label_features.json (size / contrast / SNR per label).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.algorithms.classical import ops  # noqa: E402
from gistudio.core.annotations import AnnotationStore  # noqa: E402
from gistudio.core.geometry import geometry_bbox, geometry_center  # noqa: E402
from gistudio.core.image_io import load_mono  # noqa: E402
from gistudio.core.project import Project  # noqa: E402

p = Project(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
store = AnnotationStore(p)
frames = sorted(p.all_frames(), key=lambda f: f["path"])
pos = {f["id"]: i for i, f in enumerate(frames)}
groups = defaultdict(list)
feats = []
sig_cache = {}
for f in frames:
    for a in store.for_frame(f["id"], confirmed_only=True):
        if a["label_state"] != "positive":
            continue
        img = load_mono(f["path"])
        if f["id"] not in sig_cache:
            sig_cache[f["id"]] = ops.noise_sigma(img)
        sig = sig_cache[f["id"]]
        x0, y0, x1, y1 = geometry_bbox(a["geometry_type"], a["geometry"])
        cx, cy = geometry_center(a["geometry_type"], a["geometry"])
        w, h = max(4, x1 - x0), max(4, y1 - y0)
        if a["geometry_type"] == "point":
            w = h = 6
        # appearance: inner (label region) vs. surrounding ring
        r_in = max(2.0, 0.5 * max(w, h))
        lc = ops.local_contrast_at(img, cx, cy, r_in=r_in, r_out=r_in * 2.5 + 4)
        sub = img[max(0, int(y0)):int(math.ceil(y1)) + 1, max(0, int(x0)):int(math.ceil(x1)) + 1]
        feats.append({"frame": pos[f["id"]], "ann": a["id"], "class": a["class_name"], "type": a["geometry_type"],
                      "cx": round(cx, 1), "cy": round(cy, 1), "w": round(w, 1), "h": round(h, 1),
                      "inner_mean": round(lc["inner"], 1), "ring_median": round(lc["outer"], 1),
                      "contrast": round(lc["contrast"], 1), "snr": round(lc["contrast"] / sig, 1),
                      "noise_sigma": round(sig, 2), "inner_min": float(sub.min()) if sub.size else None,
                      "saturated_nearby": bool((img[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40] >= 250).any())})
        s = int(max(64, min(400, 3 * max(w, h))))
        H, W = img.shape
        X0, Y0 = int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s))
        crop = img[Y0:Y0 + s, X0:X0 + s]
        lo, hi = np.percentile(crop, (1, 99.5))
        d = np.clip((crop.astype(np.float32) - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
        t = cv2.cvtColor(cv2.resize(d, (220, 220), interpolation=cv2.INTER_NEAREST if s <= 220 else cv2.INTER_AREA),
                         cv2.COLOR_GRAY2BGR)
        k = 220 / s
        if a["geometry_type"] == "point":
            cv2.circle(t, (int((cx - X0) * k), int((cy - Y0) * k)), int(10), (0, 0, 255), 1)
        else:
            cv2.rectangle(t, (int((x0 - X0) * k), int((y0 - Y0) * k)), (int((x1 - X0) * k), int((y1 - Y0) * k)), (0, 0, 255), 1)
        cv2.putText(t, f"f{pos[f['id']]} #{a['id']} snr {feats[-1]['snr']}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        groups[a["class_name"]].append(t)
for cls, tiles in groups.items():
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    name = cls.replace(" / ", "_").replace(" ", "_")
    ok, enc = cv2.imencode(".png", np.vstack(rows)); enc.tofile(str(out / f"labels_{name}.png"))
(out / "label_features.json").write_text(json.dumps(feats, indent=1), encoding="utf-8")
for r in feats:
    print(f"f{r['frame']:2d} {r['class']:32s} {r['type']:5s} size {r['w']:6.1f}x{r['h']:<6.1f} contrast {r['contrast']:7.1f} "
          f"snr {r['snr']:7.1f} inner {r['inner_mean']:6.1f} ring {r['ring_median']:6.1f} sat_near {r['saturated_nearby']}")
