"""Detection evaluation under partial labels.

Key rules
---------
* GT = confirmed positive annotations of defect classes only (AI candidates never count as GT).
* A prediction is a False Positive ONLY if it falls in a region verified as negative (negative ROI, hard
  negative, or a frame/class exhaustively reviewed). Predictions in unknown regions are reported as
  'unverified', never as FP. Predictions in ignore ROIs are discarded.
* Point labels are matched by distance tolerance (px), not IoU.
* Without verified OK bottles the real false-reject rate cannot be measured — reported explicitly.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from ..core import geometry as geo
from ..core.annotations import AnnotationStore
from ..core.label_targets import TargetPolicy, build_targets
from ..core.project import Project
from ..inference.engine import candidates as load_candidates

SIZE_BUCKETS = [(0, 4, "<=4px"), (4, 16, "5-16px"), (16, 64, "17-64px"), (64, 1e12, ">64px")]


@dataclass
class EvalConfig:
    tolerance_px: float = 6.0
    class_agnostic: bool = True
    thresholds: list | None = None
    operating_threshold: float | None = None
    angle_bin_deg: float = 30.0

    def to_dict(self):
        return asdict(self)


def _gt_area(a: dict, store: AnnotationStore) -> float:
    g, t = a["geometry"], a["geometry_type"]
    if "size_px" in a.get("attributes", {}):
        return float(a["attributes"]["size_px"])
    if t == "point":
        r = float(g.get("r", 1.0)); return math.pi * r * r
    if t == "bbox":
        return abs(g["x1"] - g["x0"]) * abs(g["y1"] - g["y0"])
    if t == "polygon":
        pts = np.asarray(g["points"]); x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    m = store.load_mask(a)
    return float(m.sum()) if m is not None else 0.0


def _matches(c: dict, gt: dict, tol: float) -> float | None:
    """Return distance if candidate matches GT, else None."""
    cx, cy = c["cx"], c["cy"]
    if gt["geometry_type"] == "point":
        d = math.hypot(cx - gt["gx"], cy - gt["gy"])
        return d if d <= tol else None
    x0, y0, x1, y1 = gt["bbox"]
    if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
        return math.hypot(cx - gt["gx"], cy - gt["gy"])
    return None


def collect(project: Project, run_id: int, frame_ids: list[int], cfg: EvalConfig) -> dict:
    store = AnnotationStore(project)
    cmap = project.class_map()
    policy = TargetPolicy()
    frames = {}
    for fid in frame_ids:
        f = project.frame(fid)
        shape = (f["height"], f["width"])
        anns = store.for_frame(fid, confirmed_only=True)
        gts = []
        for a in anns:
            if a["label_state"] == "positive" and cmap.get(a["class_name"], {}).get("kind") == "defect":
                gx, gy = geo.geometry_center(a["geometry_type"], a["geometry"])
                gts.append({**a, "gx": gx, "gy": gy, "bbox": geo.geometry_bbox(a["geometry_type"], a["geometry"]),
                            "area": _gt_area(a, store), "angle": f["angle_deg"]})
        target, valid, _ = build_targets(store, fid, shape, policy)
        known_neg = (valid & (target < 0.5)).any(0)
        ignore = np.zeros(shape, bool)
        for a in anns:
            if a["label_state"] == "ignore":
                ignore |= store.rasterize(a, shape)
        cands = []
        for c in load_candidates(project, run_id, fid):
            c["cx"], c["cy"] = geo.geometry_center(c["geometry_type"], c["geometry"])
            ix, iy = min(shape[1] - 1, max(0, int(c["cx"]))), min(shape[0] - 1, max(0, int(c["cy"])))
            if ignore[iy, ix]:
                continue
            c["region"] = "negative" if known_neg[iy, ix] else "unknown"
            cands.append(c)
        ms = [c["inference_ms"] for c in load_candidates(project, run_id, fid)[:1]]
        frames[fid] = {"frame": f, "gts": gts, "cands": cands, "fully_reviewed": bool(store.review_scopes(fid)),
                       "has_known_negative": bool(known_neg.any()), "ms": ms[0] if ms else None}
    return frames


def evaluate_at(frames: dict, thr: float, cfg: EvalConfig) -> dict:
    tp, fp, unverified = 0, 0, 0
    loc_err = []
    gt_hits = []  # (gt dict, hit bool)
    for fd in frames.values():
        cands = sorted([c for c in fd["cands"] if c["score"] >= thr], key=lambda c: -c["score"])
        used = set()
        matched_c = set()
        for c in cands:
            best, bd = None, None
            for gi, g in enumerate(fd["gts"]):
                if gi in used or (not cfg.class_agnostic and g["class_name"] != c["class_name"]):
                    continue
                d = _matches(c, g, cfg.tolerance_px)
                if d is not None and (bd is None or d < bd):
                    best, bd = gi, d
            if best is not None:
                used.add(best); matched_c.add(c["id"]); loc_err.append(bd)
        for c in cands:
            if c["id"] in matched_c:
                continue
            # duplicates on an already-matched GT are not FPs either
            if any(_matches(c, g, cfg.tolerance_px) is not None for g in fd["gts"]):
                continue
            if c["region"] == "negative":
                fp += 1
            else:
                unverified += 1
        tp += len(used)
        for gi, g in enumerate(fd["gts"]):
            gt_hits.append((g, gi in used))
    n_gt = len(gt_hits)
    n_frames = len(frames)
    return {"threshold": thr, "tp": tp, "fn": n_gt - tp, "fp_verified": fp, "unverified": unverified,
            "recall": tp / n_gt if n_gt else None, "precision_verified": tp / (tp + fp) if (tp + fp) else None,
            "fnr": (n_gt - tp) / n_gt if n_gt else None, "fp_per_frame": fp / n_frames if n_frames else None,
            "loc_error_mean": float(np.mean(loc_err)) if loc_err else None,
            "loc_error_p95": float(np.percentile(loc_err, 95)) if loc_err else None, "_gt_hits": gt_hits}


def _rate(hits):
    return {"n": len(hits), "recall": (sum(h for _, h in hits) / len(hits)) if hits else None}


def breakdowns(gt_hits, cfg: EvalConfig) -> dict:
    by_size, by_cls, by_ang, by_defect = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for g, h in gt_hits:
        for lo, hi, name in SIZE_BUCKETS:
            if lo < g["area"] <= hi or (lo == 0 and g["area"] <= hi):
                by_size[name].append((g, h)); break
        by_cls[g["class_name"]].append((g, h))
        if g["angle"] is not None:
            b = int(geo.wrap_deg(g["angle"]) // cfg.angle_bin_deg) * cfg.angle_bin_deg
            by_ang[f"{b:.0f}-{b + cfg.angle_bin_deg:.0f}"].append((g, h))
        if g.get("defect_id"):
            by_defect[g["defect_id"]].append(h)
    return {"by_size": {k: _rate(v) for k, v in by_size.items()},
            "by_class": {k: _rate(v) for k, v in by_cls.items()},
            "by_angle": {k: _rate(v) for k, v in sorted(by_ang.items(), key=lambda kv: float(kv[0].split('-')[0]))},
            "physical_defects": {"n": len(by_defect),
                                 "detected_in_any_view": sum(any(v) for v in by_defect.values()),
                                 "recall_any_view": (sum(any(v) for v in by_defect.values()) / len(by_defect))
                                 if by_defect else None}}


def frame_and_bottle_level(project: Project, frames: dict, thr: float) -> dict:
    fl = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "excluded": 0}
    bottles = defaultdict(lambda: {"gt_pos": False, "pred_pos": False, "product_type": ""})
    for fd in frames.values():
        f = fd["frame"]
        gt_pos = bool(fd["gts"])
        pred_pos = any(c["score"] >= thr for c in fd["cands"])
        b = bottles[f["bottle_id"]]
        b["gt_pos"] |= gt_pos; b["pred_pos"] |= pred_pos; b["product_type"] = f["product_type"]
        if not gt_pos and not fd["fully_reviewed"]:
            fl["excluded"] += 1  # cannot claim the frame is normal
            continue
        key = ("tp" if pred_pos else "fn") if gt_pos else ("fp" if pred_pos else "tn")
        fl[key] += 1
    ok_bottles = {r[0] for r in project.db.query("SELECT id FROM bottles WHERE verdict='ok'")}
    bl = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "unlabelled_bottles": 0}
    by_product = defaultdict(lambda: {"tp": 0, "fn": 0})
    for bid, b in bottles.items():
        if b["gt_pos"]:
            bl["tp" if b["pred_pos"] else "fn"] += 1
            by_product[b["product_type"]]["tp" if b["pred_pos"] else "fn"] += 1
        elif bid in ok_bottles:
            bl["fp" if b["pred_pos"] else "tn"] += 1
        else:
            bl["unlabelled_bottles"] += 1
    n_ok = bl["fp"] + bl["tn"]
    return {"frame_level": {**fl, "recall": fl["tp"] / (fl["tp"] + fl["fn"]) if fl["tp"] + fl["fn"] else None},
            "bottle_level": {**bl, "recall": bl["tp"] / (bl["tp"] + bl["fn"]) if bl["tp"] + bl["fn"] else None,
                             "false_reject_rate": (bl["fp"] / n_ok) if n_ok else None,
                             "false_reject_note": None if n_ok else
                             "NOT MEASURABLE: no verified OK bottles in the evaluated set"},
            "by_product_type": {k: {**v, "recall": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None}
                                for k, v in by_product.items()}}


def evaluate_run(project: Project, run_id: int, frame_ids: list[int], cfg: EvalConfig | None = None) -> dict:
    cfg = cfg or EvalConfig()
    frames = collect(project, run_id, frame_ids, cfg)
    scores = sorted({c["score"] for fd in frames.values() for c in fd["cands"]})
    if cfg.thresholds:
        thrs = sorted(cfg.thresholds)
    elif scores:
        thrs = sorted(set(np.quantile(scores, np.linspace(0, 1, min(40, len(scores)))).round(6).tolist()))
    else:
        thrs = [0.0]
    curve = []
    for t in thrs:
        r = evaluate_at(frames, t, cfg)
        r.pop("_gt_hits")
        curve.append(r)
    op = cfg.operating_threshold if cfg.operating_threshold is not None else thrs[0]
    at = evaluate_at(frames, op, cfg)
    hits = at.pop("_gt_hits")
    n_frames = len(frames)
    reviewed = sum(1 for fd in frames.values() if fd["fully_reviewed"])
    ms = [fd["ms"] for fd in frames.values() if fd["ms"] is not None]
    per_bottle = defaultdict(float)
    for fd in frames.values():
        if fd["ms"] is not None:
            per_bottle[fd["frame"]["bottle_id"]] += fd["ms"]
    run = project.run(run_id)
    warnings = []
    if reviewed < n_frames:
        warnings.append(f"PARTIAL LABELS: only {reviewed}/{n_frames} frames are exhaustively reviewed. Precision/FP "
                        "are computed on verified-negative regions only; 'unverified' detections may be real defects.")
    if not project.db.one("SELECT 1 FROM bottles WHERE verdict='ok' LIMIT 1"):
        warnings.append("No verified OK bottles: the production false-reject rate cannot be measured.")
    if cfg.operating_threshold is None:
        warnings.append("No operating threshold was validated; operating point shown = lowest threshold. "
                        "Do not treat any threshold as production-ready without a validation set.")
    try:
        summary = json.loads((project.root / run["output_dir"] / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    return {"run_id": run_id, "algorithm": run["algorithm"], "score_semantics": run["score_semantics"],
            "config": cfg.to_dict(), "n_frames": n_frames, "n_gt": len(hits), "label_completeness":
            {"fully_reviewed_frames": reviewed, "frames": n_frames,
             "status": "complete" if reviewed == n_frames and n_frames else "partial"},
            "operating_point": at, "froc": [{"threshold": c["threshold"], "fp_per_frame": c["fp_per_frame"],
                                             "recall": c["recall"], "unverified": c["unverified"]} for c in curve],
            "curve": curve, **breakdowns(hits, cfg), **frame_and_bottle_level(project, frames, op),
            "timing": {"frame_ms_mean": float(np.mean(ms)) if ms else None,
                       "frame_ms_max": float(np.max(ms)) if ms else None,
                       "bottle_ms_mean": float(np.mean(list(per_bottle.values()))) if per_bottle else None,
                       "gpu_memory": summary.get("model_info", {}).get("gpu_memory")},
            "warnings": warnings}


def compare_runs(project: Project, run_ids: list[int], frame_ids: list[int], cfg: EvalConfig | None = None) -> list[dict]:
    """Same test frames, same rules, multiple algorithms/models."""
    return [evaluate_run(project, r, frame_ids, cfg) for r in run_ids]


def write_report(result: dict | list[dict], path) -> None:
    from pathlib import Path
    results = result if isinstance(result, list) else [result]
    lines = ["# Glass Inspection Studio — Evaluation report", ""]
    for r in results:
        op = r["operating_point"]
        lines += [f"## Run {r['run_id']} — {r['algorithm']}", "",
                  f"* Score semantics: {r['score_semantics']}",
                  f"* Frames: {r['n_frames']}, GT instances: {r['n_gt']}, label completeness: "
                  f"{r['label_completeness']['status']} ({r['label_completeness']['fully_reviewed_frames']}/"
                  f"{r['label_completeness']['frames']} exhaustively reviewed)",
                  f"* Operating threshold: {op['threshold']:.4g} -> recall {op['recall']}, verified FP {op['fp_verified']}, "
                  f"unverified {op['unverified']}, FP/frame {op['fp_per_frame']}",
                  f"* Localization error mean/p95: {op['loc_error_mean']} / {op['loc_error_p95']} px",
                  f"* Physical defects detected in any view: {r['physical_defects']}",
                  f"* Bottle level: {r['bottle_level']}",
                  f"* Timing: {r['timing']}", "", "### Warnings", ""]
        lines += [f"* {w}" for w in r["warnings"]] + ["", "### By size", ""]
        lines += [f"* {k}: {v}" for k, v in r["by_size"].items()] + ["", "### By class", ""]
        lines += [f"* {k}: {v}" for k, v in r["by_class"].items()] + ["", "### By angle", ""]
        lines += [f"* {k}: {v}" for k, v in r["by_angle"].items()] + [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
