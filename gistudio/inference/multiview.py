"""Multi-view (rotation) aggregation to bottle level.

Principle: absence of a detection at other angles is NOT evidence of normality. No majority vote.
Bottle evidence = max over validated candidates; candidates are optionally grouped across adjacent frames
(same physical defect) — when grouping is uncertain, per-frame results are kept as separate groups.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..core import geometry as geo
from ..core.project import Project
from .engine import candidates as load_candidates


@dataclass
class AggregationConfig:
    score_threshold: float = 0.5
    min_area_px: float = 0.0
    min_local_contrast: float | None = None   # extra validation for single-frame spikes (sensor noise)
    group_max_frame_gap: int = 2              # adjacent frames considered for grouping
    group_max_dy: float = 12.0                # vertical position tolerance (px) - rotation mainly moves x
    group_max_dx: float = 80.0                # horizontal tolerance between adjacent views
    min_frames: int = 1                       # NOT forced >1: a defect may be visible at only one angle

    def to_dict(self):
        return asdict(self)


def aggregate_bottle(project: Project, run_id: int, bottle_id: int, cfg: AggregationConfig | None = None) -> dict:
    cfg = cfg or AggregationConfig()
    frames = project.all_frames([bottle_id])
    order = {f["id"]: i for i, f in enumerate(frames)}
    angle = {f["id"]: f["angle_deg"] for f in frames}
    cands = []
    for f in frames:
        for c in load_candidates(project, run_id, f["id"]):
            if c["score"] < cfg.score_threshold or (c["area_px"] or 0) < cfg.min_area_px:
                continue
            lc = c["features"].get("local_contrast")
            if cfg.min_local_contrast is not None and lc is not None and lc < cfg.min_local_contrast:
                continue
            c["cx"], c["cy"] = geo.geometry_center(c["geometry_type"], c["geometry"])
            c["order"] = order[f["id"]]
            cands.append(c)
    # greedy grouping across nearby frames (union-find); ambiguous matches (2+ in tolerance) are not merged
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_frames = len(frames)
    for i, a in enumerate(cands):
        matches = []
        for j, b in enumerate(cands):
            if j <= i or a["class_name"] != b["class_name"] or a["frame_id"] == b["frame_id"]:
                continue
            gap = abs(a["order"] - b["order"])
            gap = min(gap, n_frames - gap)  # circular sequence
            if 0 < gap <= cfg.group_max_frame_gap and abs(a["cy"] - b["cy"]) <= cfg.group_max_dy \
                    and abs(a["cx"] - b["cx"]) <= cfg.group_max_dx * gap:
                matches.append(j)
        by_frame = {}
        for j in matches:
            by_frame.setdefault(cands[j]["frame_id"], []).append(j)
        for fid, js in by_frame.items():
            if len(js) == 1:  # unambiguous
                parent[find(js[0])] = find(i)
    groups: dict[int, list[dict]] = {}
    for i, c in enumerate(cands):
        groups.setdefault(find(i), []).append(c)
    out_groups = []
    for gi, members in enumerate(groups.values()):
        if len({m["frame_id"] for m in members}) < cfg.min_frames:
            continue
        best = max(members, key=lambda m: m["score"])
        out_groups.append({"group": gi, "class_name": best["class_name"], "max_score": best["score"],
                           "best_view_frame": best["frame_id"], "best_view_angle": angle.get(best["frame_id"]),
                           "best_candidate_id": best["id"], "n_views": len({m["frame_id"] for m in members}),
                           "angles": sorted({angle.get(m["frame_id"]) for m in members if angle.get(m["frame_id"]) is not None}),
                           "candidate_ids": [m["id"] for m in members]})
        key = f"run{run_id}-b{bottle_id}-g{gi}"
        for m in members:
            project.db.execute("UPDATE candidates SET group_key=? WHERE id=?", (key, m["id"]))
    project.db.commit()
    out_groups.sort(key=lambda g: -g["max_score"])
    return {"bottle_id": bottle_id, "run_id": run_id, "config": cfg.to_dict(),
            "bottle_max_score": out_groups[0]["max_score"] if out_groups else 0.0,
            "decision": "NG-candidate" if out_groups else "no-candidate-above-threshold",
            "note": "no-candidate does NOT prove OK (unobserved angles / undetectable defects)",
            "groups": out_groups}


def save_evidence(project: Project, agg: dict, out_dir, crop: int = 128) -> list[str]:
    """Save best-view crops (display-normalised PNG) as NG evidence."""
    from pathlib import Path
    from ..core.image_io import load_mono, write_image_lossless
    from ..core.preprocessing import DisplayTransform
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for g in agg["groups"]:
        c = project.db.one("SELECT * FROM candidates WHERE id=?", (g["best_candidate_id"],))
        f = project.frame(c["frame_id"])
        img = load_mono(f["path"])
        cx, cy = geo.geometry_center(c["geometry_type"], json.loads(c["geometry"]))
        h, w = img.shape
        x0, y0 = int(max(0, cx - crop // 2)), int(max(0, cy - crop // 2))
        patch = img[y0:min(h, y0 + crop), x0:min(w, x0 + crop)]
        disp = DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=f.get("bit_depth")).apply(patch)
        p = out_dir / f"b{agg['bottle_id']}_g{g['group']}_f{f['id']}_a{(f['angle_deg'] or 0):.1f}.png"
        write_image_lossless(p, disp)
        paths.append(str(p))
    (out_dir / f"bottle_{agg['bottle_id']}_aggregation.json").write_text(json.dumps(agg, indent=2, default=str),
                                                                        encoding="utf-8")
    return paths
