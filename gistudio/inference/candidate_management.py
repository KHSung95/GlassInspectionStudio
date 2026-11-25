"""Active-learning queues over candidates. Candidates stay out of ground truth until reviewed."""
from __future__ import annotations

import math
import random

import numpy as np

from ..core import geometry as geo
from ..core.annotations import AnnotationStore
from ..core.project import Project
from .engine import candidates as load_candidates

STRATEGIES = ("high_score", "uncertain", "label_disagreement", "on_normal_structure", "new_unlabelled",
              "model_disagreement", "random")


def _ann_index(store: AnnotationStore, frame_id: int):
    out = []
    for a in store.for_frame(frame_id):
        out.append((a, geo.geometry_bbox(a["geometry_type"], a["geometry"])))
    return out


def _near(c, idx, pad=4.0):
    cx, cy = geo.geometry_center(c["geometry_type"], c["geometry"])
    return [a for a, (x0, y0, x1, y1) in idx if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad]


def review_queue(project: Project, run_id: int, strategy: str = "high_score", threshold: float = 0.5,
                 other_run_id: int | None = None, limit: int = 200, include_reviewed: bool = False,
                 seed: int = 0, tol_px: float = 6.0) -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    store = AnnotationStore(project)
    cmap = project.class_map()
    cands = load_candidates(project, run_id, review_status=None if include_reviewed else "pending")
    idx_cache = {}

    def idx(fid):
        if fid not in idx_cache:
            idx_cache[fid] = _ann_index(store, fid)
        return idx_cache[fid]

    if strategy == "high_score":
        out = sorted(cands, key=lambda c: -c["score"])
    elif strategy == "uncertain":
        out = sorted(cands, key=lambda c: abs(c["score"] - threshold))
    elif strategy == "label_disagreement":
        out = []
        for c in cands:
            near = _near(c, idx(c["frame_id"]))
            if any(a["label_state"] == "negative" for a in near) and c["score"] >= threshold:
                c["reason"] = "high score on verified-negative label"; out.append(c)
        # missed positives: GT positives with no candidate nearby -> surfaced as pseudo-items
        out.sort(key=lambda c: -c["score"])
    elif strategy == "on_normal_structure":
        out = [c for c in cands if any(cmap.get(a["class_name"], {}).get("kind") == "normal"
                                       for a in _near(c, idx(c["frame_id"]), pad=8))]
        out.sort(key=lambda c: -c["score"])
    elif strategy == "new_unlabelled":
        out = [c for c in cands if c["score"] >= threshold and not _near(c, idx(c["frame_id"]))]
        out.sort(key=lambda c: -c["score"])
    elif strategy == "model_disagreement":
        if other_run_id is None:
            raise ValueError("model_disagreement requires other_run_id")
        others = {}
        for o in load_candidates(project, other_run_id):
            others.setdefault(o["frame_id"], []).append(geo.geometry_center(o["geometry_type"], o["geometry"]))
        out = []
        for c in cands:
            cx, cy = geo.geometry_center(c["geometry_type"], c["geometry"])
            if c["score"] >= threshold and not any(math.hypot(cx - x, cy - y) <= tol_px
                                                   for x, y in others.get(c["frame_id"], [])):
                c["reason"] = f"not found by run {other_run_id}"; out.append(c)
        out.sort(key=lambda c: -c["score"])
    else:
        out = list(cands)
        random.Random(seed).shuffle(out)
    return out[:limit]


def missed_positives(project: Project, run_id: int, frame_ids: list[int], tol_px: float = 6.0,
                     threshold: float = 0.0) -> list[dict]:
    """Confirmed positive labels without any candidate nearby (model/label disagreement of the FN kind)."""
    store = AnnotationStore(project)
    cmap = project.class_map()
    out = []
    for fid in frame_ids:
        pts = [geo.geometry_center(c["geometry_type"], c["geometry"]) for c in load_candidates(project, run_id, fid)
               if c["score"] >= threshold]
        for a in store.for_frame(fid, confirmed_only=True):
            if a["label_state"] == "positive" and cmap.get(a["class_name"], {}).get("kind") == "defect":
                x, y = geo.geometry_center(a["geometry_type"], a["geometry"])
                if not any(math.hypot(x - px, y - py) <= tol_px for px, py in pts):
                    out.append(a)
    return out


def apply_group_review(project: Project, candidate_id: int, decision: str, class_name: str | None = None,
                       author: str = "") -> list[int]:
    """Apply one review decision to every candidate in the same multi-view group (same physical defect).
    Accepted members are linked to a common Defect ID."""
    store = AnnotationStore(project, author)
    c = project.db.one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
    members = [candidate_id]
    if c["group_key"]:
        members = [r[0] for r in project.db.query("SELECT id FROM candidates WHERE group_key=? AND review_status='pending'",
                                                  (c["group_key"],))] or [candidate_id]
    created = []
    defect_id = None
    if decision == "accept":
        bottle_id = project.frame(c["frame_id"])["bottle_id"]
        defect_id = store.create_defect(bottle_id, class_name or c["class_name"])
    for mid in members:
        if decision == "accept":
            created.append(store.accept_candidate(mid, class_name, defect_id))
        elif decision == "reject":
            a = store.reject_candidate(mid)
            if a:
                created.append(a)
        elif decision == "unknown":
            store.mark_candidate_unknown(mid)
        else:
            raise ValueError(decision)
    return created
