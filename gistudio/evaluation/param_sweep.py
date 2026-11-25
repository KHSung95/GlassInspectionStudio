"""Label-driven parameter sweep for classical detectors → several recommended presets.

How it works
------------
1. Ground truth = CONFIRMED positive labels of the classes the detector targets, on the chosen frames.
   Verified-negative regions (negative ROIs, hard negatives, exhaustively reviewed frames) define where a
   detection counts as a false positive; detections elsewhere are 'unverified' (partial labels!).
2. For speed, each frame is processed on crops around its labels (+margin) unless the frame was exhaustively
   reviewed (then the full frame). The frame-level noise sigma is passed in, so SNR scores and thresholds are
   identical to full-frame runs.
3. Every parameter combination is run at every threshold of a grid. The threshold is NOT swept analytically
   from one low-threshold run, because connected components (candidate positions) change with the threshold:
   at low thresholds a small spot merges with nearby ink/edges and its candidate moves away (observed on real data).
4. Several objectives pick operating points (balanced F1, high recall, low FP, zero verified FP). Optionally a
   bottle-level hold-out set reports the same metrics on unseen bottles to expose over-fitting.

The result is a recommendation from a (usually small) labelled set — it is NOT a validated production setting.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..algorithms.base import create, registry
from ..algorithms.classical import ops
from ..core import geometry as geo
from ..core.annotations import AnnotationStore
from ..core.image_io import load_mono
from ..core.label_targets import TargetPolicy, build_targets
from ..core.project import Project, now_iso
from .metrics import EvalConfig, evaluate_at

SPOT_CLASSES = ["Black Spot"]
BUBBLE_CLASSES = ["Bubble", "Open Bubble", "Closed Bubble"]
AREA_CLASSES = ["Intensity / Transparency Defect"]

# Explicit, physically sensible grids (threshold handled analytically, so it is not in the grid).
SPACES: dict[str, dict[str, list]] = {
    "classical.blackhat": {"kernel": [5, 9, 15, 21], "pre_sigma": [0.0, 1.2], "min_area": [1, 2], "max_area": [60, 400]},
    "classical.dog": {"sigma_small": [0.7, 1.0, 1.5], "sigma_large": [3.0, 4.0, 6.0, 8.0], "max_area": [60, 400]},
    "classical.log_blob": {"sigma_min": [0.6, 0.8, 1.2], "sigma_max": [2.0, 3.0, 5.0], "n_scales": [3], "max_area": [60, 400]},
    "classical.local_background": {"bg_kernel": [15, 31, 61], "downsample": [1, 2], "pre_sigma": [0.0, 0.8]},
    "classical.local_contrast": {"window": [9, 15, 25, 41], "pre_sigma": [0.0, 0.8]},
    "classical.adaptive_threshold": {"window": [15, 25, 51], "offset_sigma": [3.0, 4.0, 5.0, 6.0]},
    "classical.lowfreq_deviation": {"scales": ["2,4,8", "4,8,16", "8,16,32"], "context_factor": [3.0, 4.0, 6.0],
                                    "min_area": [20, 50, 100]},
    "classical.local_intensity_deviation": {"smooth_sigma": [2.0, 3.0, 5.0], "bg_kernel": [61, 101, 201],
                                            "min_area": [20, 50, 100]},
    "classical.gradient_texture": {"window": [9, 15, 25], "context": [101, 151, 301], "min_area": [20, 30, 60]},
    "classical.bubble_contour": {"sigma_min": [1.0, 2.0], "sigma_max": [6.0, 8.0, 12.0],
                                 "min_circularity": [0.3, 0.5, 0.7]},
}
DEFAULT_THRESHOLD_GRID = [4.0, 6.0, 8.0, 11.0, 16.0, 22.0, 32.0, 45.0, 64.0]
THRESHOLD_GRID = {"classical.local_contrast": [4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
                  "classical.adaptive_threshold": [3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
                  "classical.gradient_texture": [1.5, 2.0, 3.0, 4.0, 6.0],
                  "classical.lowfreq_deviation": [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                  "classical.local_intensity_deviation": [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                  "classical.bubble_contour": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]}


def default_gt_classes(algorithm_id: str) -> list[str]:
    cls = registry()[algorithm_id].default_class
    if cls == "Black Spot":
        return SPOT_CLASSES
    if cls == "Bubble":
        return BUBBLE_CLASSES
    if cls.startswith("Intensity"):
        return AREA_CLASSES
    return SPOT_CLASSES + BUBBLE_CLASSES + AREA_CLASSES


@dataclass
class SweepConfig:
    algorithm: str
    frame_ids: list
    space: dict | None = None
    gt_classes: list | None = None
    tolerance_px: float = 6.0
    crop_margin: int = 160            # 0 = always full frame
    max_combos: int = 60
    seed: int = 0
    recall_target: float = 0.9
    fp_budget_per_frame: float = 0.2
    holdout_fraction: float = 0.0     # bottle-level hold-out for validation of the recommendations
    max_candidates: int = 3000
    min_useful_recall: float = 0.3    # low-FP / zero-FP presets below this recall are not recommended
    threshold_grid: list | None = None  # None -> per-algorithm default grid

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameTruth:
    frame: dict
    gts: list
    known_neg: np.ndarray
    ignore: np.ndarray
    rects: list                        # crops (x0, y0, x1, y1) in original coordinates
    noise_sigma: float
    fully_reviewed: bool


def _merge_rects(rects: list, w: int, h: int) -> list:
    rs = [[max(0, int(a)), max(0, int(b)), min(w, int(math.ceil(c))), min(h, int(math.ceil(d)))] for a, b, c, d in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rs:
            r = rs.pop()
            for o in out:
                if not (r[2] <= o[0] or o[2] <= r[0] or r[3] <= o[1] or o[3] <= r[1]):
                    o[0], o[1], o[2], o[3] = min(o[0], r[0]), min(o[1], r[1]), max(o[2], r[2]), max(o[3], r[3])
                    changed = True
                    break
            else:
                out.append(r)
        rs = out
    return [tuple(r) for r in rs if r[2] > r[0] and r[3] > r[1]]


def frame_truth(project: Project, fid: int, gt_classes: list[str], margin: int) -> FrameTruth:
    store = AnnotationStore(project)
    f = project.frame(fid)
    shape = (f["height"], f["width"])
    anns = store.for_frame(fid, confirmed_only=True)
    gts = []
    for a in anns:
        if a["label_state"] == "positive" and a["class_name"] in gt_classes:
            gx, gy = geo.geometry_center(a["geometry_type"], a["geometry"])
            gts.append({**a, "gx": gx, "gy": gy, "bbox": geo.geometry_bbox(a["geometry_type"], a["geometry"]),
                        "area": 0.0, "angle": f["angle_deg"]})
    target, valid, _ = build_targets(store, fid, shape, TargetPolicy())
    known_neg = (valid & (target < 0.5)).any(0)
    ignore = np.zeros(shape, bool)
    for a in anns:
        if a["label_state"] == "ignore":
            ignore |= store.rasterize(a, shape)
    reviewed = bool(store.review_scopes(fid))
    if reviewed or margin <= 0:
        rects = [(0, 0, f["width"], f["height"])]
    else:
        rects = []
        for a in anns:
            if a["label_state"] in ("positive", "negative"):
                x0, y0, x1, y1 = geo.geometry_bbox(a["geometry_type"], a["geometry"])
                rects.append((x0 - margin, y0 - margin, x1 + margin, y1 + margin))
        rects = _merge_rects(rects, f["width"], f["height"])
    img = load_mono(f["path"])
    return FrameTruth(f, gts, known_neg, ignore, rects, ops.noise_sigma(img), reviewed)


def _combos(space: dict, max_combos: int, seed: int) -> list[dict]:
    keys = list(space)
    allc = [dict(zip(keys, v)) for v in itertools.product(*[space[k] for k in keys])]
    if len(allc) <= max_combos:
        return allc
    rnd = random.Random(seed)
    return rnd.sample(allc, max_combos)


def _detect(alg_id: str, params: dict, truths: list[FrameTruth], max_cands: int) -> tuple[dict, float]:
    alg = create(alg_id, **params)
    frames, t0 = {}, time.perf_counter()
    cid = 0
    for t in truths:
        img = load_mono(t.frame["path"])
        cands = []
        for (x0, y0, x1, y1) in t.rects:
            res = alg.run(img[y0:y1, x0:x1], {"noise_sigma": t.noise_sigma, "bit_depth": t.frame.get("bit_depth")})
            for c in res.candidates:
                cx, cy = geo.geometry_center(c.geometry_type, c.geometry)
                cx, cy = cx + x0, cy + y0
                ix, iy = min(t.frame["width"] - 1, int(cx)), min(t.frame["height"] - 1, int(cy))
                if t.ignore[iy, ix]:
                    continue
                cid += 1
                cands.append({"id": cid, "score": float(c.score), "cx": cx, "cy": cy, "class_name": c.class_name,
                              "region": "negative" if t.known_neg[iy, ix] else "unknown"})
        cands.sort(key=lambda d: -d["score"])
        frames[t.frame["id"]] = {"frame": t.frame, "gts": t.gts, "cands": cands[:max_cands],
                                 "fully_reviewed": t.fully_reviewed}
    return frames, (time.perf_counter() - t0) * 1000 / max(1, len(truths))


def _metrics(frames: dict, thr: float, tol: float) -> dict:
    r = evaluate_at(frames, thr, EvalConfig(tolerance_px=tol))
    r.pop("_gt_hits", None)
    n = max(1, len(frames))
    p = r["precision_verified"] if r["precision_verified"] is not None else (1.0 if r["tp"] else 0.0)
    rec = r["recall"] or 0.0
    r["f1_verified"] = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    r["unverified_per_frame"] = r["unverified"] / n
    return r


def _thresholds(frames: dict, n: int = 40) -> list[float]:
    s = sorted({c["score"] for fd in frames.values() for c in fd["cands"]})
    if not s:
        return []
    q = np.unique(np.quantile(s, np.linspace(0, 1, min(n, len(s)))))
    return [float(v) for v in q]


def _pick(trials: list[dict], cfg: SweepConfig) -> dict[str, dict]:
    """trials: one entry per (combo, threshold) with 'm' metrics."""
    def key_unv(t):
        return (t["m"]["fp_per_frame"] or 0, t["m"]["unverified_per_frame"])
    out = {}
    if not trials:
        return out
    out["balanced"] = max(trials, key=lambda t: (t["m"]["f1_verified"], -t["m"]["unverified_per_frame"]))
    hr = [t for t in trials if (t["m"]["recall"] or 0) >= cfg.recall_target]
    out["high_recall"] = (min(hr, key=key_unv) if hr else
                          max(trials, key=lambda t: ((t["m"]["recall"] or 0), -key_unv(t)[0], -key_unv(t)[1])))
    useful = [t for t in trials if (t["m"]["recall"] or 0) >= cfg.min_useful_recall]
    lf = [t for t in useful if (t["m"]["fp_per_frame"] or 0) <= cfg.fp_budget_per_frame]
    if lf:
        out["low_fp"] = max(lf, key=lambda t: ((t["m"]["recall"] or 0), -t["m"]["unverified_per_frame"]))
    zf = [t for t in useful if t["m"]["fp_verified"] == 0]
    if zf:
        out["zero_verified_fp"] = max(zf, key=lambda t: ((t["m"]["recall"] or 0), -t["m"]["unverified_per_frame"]))
    return out


DESCRIPTIONS = {
    "balanced": "max F1 (precision on verified-negative areas × recall)",
    "high_recall": "recall ≥ target with the fewest false/unverified detections",
    "low_fp": "verified FP/frame ≤ budget with the highest recall",
    "zero_verified_fp": "no detection on verified-normal areas, highest recall",
}


def run_sweep(project: Project, cfg: SweepConfig, progress=None, stop=None) -> dict:
    alg_cls = registry()[cfg.algorithm]
    if alg_cls.family != "classical":
        raise ValueError("parameter sweep is intended for classical detectors")
    space = dict(cfg.space or SPACES.get(cfg.algorithm) or {})
    gt_classes = cfg.gt_classes or default_gt_classes(cfg.algorithm)
    # bottle-level hold-out
    fr = [project.frame(f) for f in cfg.frame_ids]
    bottles = sorted({f["bottle_id"] for f in fr})
    hold: set = set()
    warnings = []
    if cfg.holdout_fraction > 0:
        if len(bottles) >= 2:
            rnd = random.Random(cfg.seed)
            k = max(1, int(round(len(bottles) * cfg.holdout_fraction)))
            hold = set(rnd.sample(bottles, k))
        else:
            warnings.append("hold-out requested but only one bottle has labels: no independent validation")
    tune_ids = [f["id"] for f in fr if f["bottle_id"] not in hold]
    hold_ids = [f["id"] for f in fr if f["bottle_id"] in hold]
    truths = [frame_truth(project, f, gt_classes, cfg.crop_margin) for f in tune_ids]
    truths = [t for t in truths if t.gts or t.known_neg.any()]
    n_gt = sum(len(t.gts) for t in truths)
    if n_gt == 0:
        raise RuntimeError(f"No confirmed positive labels of {gt_classes} on the selected frames. Label a few first.")
    hold_truths = [frame_truth(project, f, gt_classes, cfg.crop_margin) for f in hold_ids]
    n_neg_frames = sum(1 for t in truths if t.known_neg.any())
    if n_neg_frames == 0:
        warnings.append("No verified-negative regions: false positives cannot be measured, precision is optimistic. "
                        "Add Normal ROIs (O) over gratings/digits/edges.")
    if n_gt < 10:
        warnings.append(f"Only {n_gt} labelled defects: recommendations may over-fit.")
    combos = _combos(space, cfg.max_combos, cfg.seed)
    base = {p.name: p.default for p in alg_cls.params_spec}
    # The threshold is a real detection parameter: connected components (and therefore candidate positions)
    # change with it — at a low threshold a tiny spot merges with neighbouring ink/edges. So every threshold in
    # the grid is detected separately and evaluated only at its own value (no analytic re-thresholding).
    grid = list(cfg.threshold_grid or THRESHOLD_GRID.get(cfg.algorithm, DEFAULT_THRESHOLD_GRID))
    trials, per_combo = [], []
    t_start = time.time()
    total = len(combos) * len(grid)
    step = 0
    for i, combo in enumerate(combos):
        best = None
        ms_list = []
        for thr in grid:
            if stop and stop():
                break
            params = {**base, **combo, "threshold": thr}
            if "max_candidates" in params:
                params["max_candidates"] = cfg.max_candidates
            frames, ms = _detect(cfg.algorithm, params, truths, cfg.max_candidates)
            ms_list.append(ms)
            m = _metrics(frames, thr, cfg.tolerance_px)
            t = {"combo": combo, "threshold": thr, "m": m, "ms_frame_crops": ms}
            trials.append(t)
            if best is None or m["f1_verified"] > best["m"]["f1_verified"]:
                best = t
            step += 1
            if progress:
                progress(step, total, f"{combo} thr={thr} recall={m['recall']} FPver={m['fp_verified']}")
        if stop and stop():
            warnings.append("sweep stopped early by the user")
            break
        ms = float(np.mean(ms_list)) if ms_list else 0.0
        per_combo.append({"combo": combo, "best_f1": best["m"]["f1_verified"] if best else 0.0,
                          "best_threshold": best["threshold"] if best else None, "ms": ms})
    picks = _pick(trials, cfg)
    for name in ("low_fp", "zero_verified_fp"):
        if name not in picks:
            warnings.append(f"'{name}' not recommended: no setting met the FP condition with recall ≥ "
                            f"{cfg.min_useful_recall} (detector cannot separate defects from labelled normals).")
    recs = []
    for name, t in picks.items():
        params = {**base, **t["combo"], "threshold": round(t["threshold"], 4)}
        rec = {"objective": name, "description": DESCRIPTIONS[name], "params": params, "tune": _short(t["m"])}
        if hold_truths:
            frames, _ = _detect(cfg.algorithm, {**params, "max_candidates": cfg.max_candidates}, hold_truths,
                                cfg.max_candidates)
            rec["holdout"] = _short(_metrics(frames, t["threshold"], cfg.tolerance_px))
        recs.append(rec)
    # baseline (defaults) for comparison — detected at its own threshold
    bparams = {**base, **({"max_candidates": cfg.max_candidates} if "max_candidates" in base else {})}
    frames, _ = _detect(cfg.algorithm, bparams, truths, cfg.max_candidates)
    baseline = {"params": base, "tune": _short(_metrics(frames, base.get("threshold", 0.0), cfg.tolerance_px))}
    return {"algorithm": cfg.algorithm, "config": cfg.to_dict(), "gt_classes": gt_classes, "n_gt": n_gt,
            "tune_frames": len(truths), "holdout_frames": len(hold_truths), "verified_negative_frames": n_neg_frames,
            "combos_evaluated": len(per_combo), "seconds": round(time.time() - t_start, 1),
            "recommendations": recs, "baseline_defaults": baseline,
            "top_combos": sorted(per_combo, key=lambda c: -c["best_f1"])[:15], "warnings": warnings,
            "note": "Recommended from partially labelled data; validate on independent bottles before use."}


def _short(m: dict) -> dict:
    keys = ("threshold", "tp", "fn", "fp_verified", "unverified", "recall", "precision_verified", "f1_verified",
            "fp_per_frame", "unverified_per_frame", "loc_error_mean")
    return {k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in keys}


def save_presets(project: Project, result: dict, prefix: str = "sweep") -> list[str]:
    d = project.root / "presets"
    d.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths = []
    for r in result["recommendations"]:
        name = f"{prefix}_{result['algorithm'].split('.')[-1]}_{r['objective']}_{stamp}"
        payload = {"algorithm": result["algorithm"], "params": r["params"], "objective": r["objective"],
                   "description": r["description"], "metrics_tune": r["tune"], "metrics_holdout": r.get("holdout"),
                   "n_gt": result["n_gt"], "gt_classes": result["gt_classes"], "created_at": now_iso(),
                   "note": result["note"]}
        p = d / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        paths.append(str(p))
    run = project.create_run("eval", f"sweep:{result['algorithm']}", params=result["config"],
                             notes=f"param sweep, {len(paths)} presets")
    (run["output_dir"] / "sweep_result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    project.finish_run(run["id"], "done")
    return paths


def labelled_frames(project: Project, bottle_ids: list[int] | None = None) -> list[int]:
    q = """SELECT DISTINCT a.frame_id FROM annotations a JOIN frames f ON a.frame_id=f.id JOIN sessions s ON f.session_id=s.id
           WHERE a.deleted=0 AND a.confirmed=1 AND a.label_state IN ('positive','negative')"""
    p: list = []
    if bottle_ids:
        q += f" AND s.bottle_id IN ({','.join('?' * len(bottle_ids))})"; p = list(bottle_ids)
    return [r[0] for r in project.db.query(q + " ORDER BY a.frame_id", p)]
