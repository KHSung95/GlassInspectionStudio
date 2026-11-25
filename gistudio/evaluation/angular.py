"""Angular Visibility Analysis.

For each physical defect (Defect ID) we measure, per captured angle:
  * detection score of a run (max candidate score within tolerance of the annotated location)
  * local contrast (annulus median - disc mean), brightness, approximate apparent size
and derive best view, visible angular intervals (circular), and a software re-sampling experiment that
emulates coarser capture steps / different start angles to estimate miss rates (-> camera FPS / rotation
speed requirements).

Frames where the defect has no annotation are treated as 'unknown location' and therefore NOT detected in
the sampling experiment (conservative), and reported in `coverage`.
"""
from __future__ import annotations

import math

import numpy as np

from ..algorithms.classical import ops
from ..core import geometry as geo
from ..core.annotations import AnnotationStore
from ..core.image_io import load_mono
from ..core.project import Project
from ..inference.engine import candidates as load_candidates


def apparent_size(img: np.ndarray, x: float, y: float, r_out: float = 8.0, k: float = 3.0) -> float:
    lc = ops.local_contrast_at(img, x, y, 1.0, r_out)
    thr = lc["outer"] - k * max(lc.get("outer_std", 0.0), 1e-6)
    R = int(r_out)
    h, w = img.shape
    x0, y0 = max(0, int(x) - R), max(0, int(y) - R)
    patch = img[y0:min(h, int(y) + R + 1), x0:min(w, int(x) + R + 1)].astype(np.float32)
    if patch.size == 0:
        return 0.0
    import cv2
    m = (patch < thr).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    cy, cx = int(y) - y0, int(x) - x0
    if 0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1] and lab[cy, cx] > 0:
        return float((lab == lab[cy, cx]).sum())
    return 0.0


def defect_profile(project: Project, defect_id: int, run_id: int | None = None, tol_px: float = 6.0) -> dict:
    store = AnnotationStore(project)
    d = project.db.one("SELECT * FROM defects WHERE id=?", (defect_id,))
    obs = {o["frame_id"]: o for o in store.defect_observations(defect_id)}
    frames = project.all_frames([d["bottle_id"]])
    rows = []
    for f in frames:
        o = obs.get(f["id"])
        row = {"frame_id": f["id"], "angle": f["angle_deg"], "session": f["session_code"], "annotated": o is not None}
        if o is not None:
            x, y = geo.geometry_center(o["geometry_type"], o["geometry"])
            img = load_mono(f["path"])
            lc = ops.local_contrast_at(img, x, y)
            row.update({"x": x, "y": y, "visibility": o["visibility"], "local_contrast": lc["contrast"],
                        "brightness": lc["inner"], "size_px": apparent_size(img, x, y)})
            if run_id is not None:
                best = 0.0
                for c in load_candidates(project, run_id, f["id"]):
                    cx, cy = geo.geometry_center(c["geometry_type"], c["geometry"])
                    if math.hypot(cx - x, cy - y) <= tol_px:
                        best = max(best, c["score"])
                row["score"] = best
        rows.append(row)
    ann = [r for r in rows if r["annotated"] and r["angle"] is not None]
    best_score = max(ann, key=lambda r: r.get("score", -1), default=None) if run_id is not None else None
    best_contrast = max(ann, key=lambda r: r["local_contrast"], default=None)
    return {"defect_id": defect_id, "defect_uid": d["defect_uid"], "class_name": d["class_name"], "rows": rows,
            "best_view_by_score": best_score and {"frame_id": best_score["frame_id"], "angle": best_score["angle"],
                                                  "score": best_score.get("score")},
            "best_view_by_contrast": best_contrast and {"frame_id": best_contrast["frame_id"],
                                                        "angle": best_contrast["angle"],
                                                        "contrast": best_contrast["local_contrast"]},
            "coverage": {"frames": len(rows), "annotated": len(ann)}}


def visible_intervals(profile: dict, key: str = "score", thr: float = 0.5, session: str | None = None) -> list:
    rows = [r for r in profile["rows"] if r["angle"] is not None and (session is None or r["session"] == session)]
    rows.sort(key=lambda r: geo.wrap_deg(r["angle"]))
    if len(rows) < 2:
        return []
    angles = [geo.wrap_deg(r["angle"]) for r in rows]
    step = float(np.median(np.diff(angles)))
    if key == "visibility":
        vis = [r.get("visibility") in ("visible", "weak") for r in rows]
    else:
        vis = [r.get(key, 0.0) is not None and r.get(key, 0.0) >= thr and r["annotated"] for r in rows]
    return geo.circular_intervals(angles, vis, step)


def sampling_experiment(profiles: list[dict], steps=(1, 2, 5, 10, 15, 30, 90), n_starts: int = 12,
                        key: str = "score", thr: float = 0.5) -> dict:
    """Re-sample dense captures at coarser steps and multiple start angles; report miss rates."""
    out = {}
    for step in steps:
        misses, worst = [], 0.0
        per_start = []
        starts = np.linspace(0, step, n_starts, endpoint=False)
        for s0 in starts:
            missed = 0
            for prof in profiles:
                rows = [r for r in prof["rows"] if r["angle"] is not None]
                if not rows:
                    continue
                angles = np.array([geo.wrap_deg(r["angle"]) for r in rows])
                native = float(np.median(np.diff(np.sort(angles)))) if len(angles) > 1 else 360.0
                if step < native - 1e-9:
                    continue
                det = np.array([(r.get("visibility") in ("visible", "weak")) if key == "visibility"
                                else (r["annotated"] and (r.get(key) or 0) >= thr) for r in rows])
                k = int(math.ceil(360.0 / step))
                hit = False
                for i in range(k):
                    a = geo.wrap_deg(s0 + i * step)
                    j = geo.nearest_angle_index(angles, a)
                    if geo.circ_dist(angles[j], a) <= native / 2 + 1e-6 and det[j]:
                        hit = True; break
                missed += (not hit)
            n = len(profiles)
            rate = missed / n if n else None
            per_start.append({"start": float(s0), "miss_rate": rate})
            if rate is not None:
                misses.append(rate); worst = max(worst, rate)
        out[f"{float(step):g}"] = {"mean_miss_rate": float(np.mean(misses)) if misses else None, "worst_miss_rate": worst,
                          "frames_per_rev": int(math.ceil(360 / step)), "per_start": per_start}
    return out


def required_capture(step_deg: float, rev_time_s: float) -> dict:
    """Capture rate implied by an angular step for a given time per revolution."""
    n = 360.0 / step_deg
    return {"frames_per_rev": n, "fps": n / rev_time_s, "deg_per_s": 360.0 / rev_time_s}


def repeatability(project: Project, defect_id: int, run_id: int) -> dict:
    """Compare per-angle scores across sessions (repeat captures) of the same bottle."""
    prof = defect_profile(project, defect_id, run_id)
    by_s: dict[str, dict[float, float]] = {}
    for r in prof["rows"]:
        if r["annotated"] and r["angle"] is not None:
            by_s.setdefault(r["session"], {})[round(geo.wrap_deg(r["angle"]), 1)] = r.get("score", 0.0)
    ss = sorted(by_s)
    res = {}
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            common = sorted(set(by_s[ss[i]]) & set(by_s[ss[j]]))
            if len(common) >= 3:
                a = np.array([by_s[ss[i]][k] for k in common]); b = np.array([by_s[ss[j]][k] for k in common])
                corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
                res[f"{ss[i]}~{ss[j]}"] = {"n": len(common), "pearson": corr, "mean_abs_diff": float(np.abs(a - b).mean())}
    return res
