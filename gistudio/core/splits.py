"""Bottle-level (physical sample) dataset splits — prevents leakage of the same bottle across subsets."""
from __future__ import annotations

import json
import random
from collections import defaultdict

from .project import Project, now_iso


def make_split(project: Project, name: str, ratios=(0.6, 0.2, 0.2), seed: int = 0,
               dataset_id: int | None = None, holdout_product_types: list[str] | None = None,
               stratify_by_product: bool = True) -> int:
    """Assign every *physical sample* (all bottles sharing physical_sample_id) to exactly one subset.

    holdout_product_types: all bottles of those product types go to 'test' (leave-product-out generalisation).
    """
    if project.db.one("SELECT id FROM splits WHERE name=?", (name,)):
        raise ValueError(f"Split '{name}' already exists")
    bottles = project.bottles(dataset_id)
    holdout = set(holdout_product_types or [])
    groups: dict[str, list[dict]] = defaultdict(list)
    for b in bottles:
        groups[b["physical_sample_id"] or b["bottle_code"]].append(b)
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for gid, bs in sorted(groups.items()):
        if any(b["product_type"] in holdout for b in bs):
            assign[gid] = "test"
        else:
            strata[bs[0]["product_type"] if stratify_by_product else "_"].append(gid)
    for _, gids in sorted(strata.items()):
        rng.shuffle(gids)
        n = len(gids)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        if n >= 3 and ratios[1] > 0 and n_va == 0:
            n_va = 1
        if n >= 3 and ratios[2] > 0 and n - n_tr - n_va <= 0:
            n_tr = max(1, n_tr - 1)
        for i, g in enumerate(gids):
            assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    cur = project.db.execute("INSERT INTO splits(name, seed, config, created_at) VALUES(?,?,?,?)",
                             (name, seed, json.dumps({"ratios": ratios, "holdout_product_types": sorted(holdout),
                                                      "dataset_id": dataset_id,
                                                      "stratify_by_product": stratify_by_product}), now_iso()))
    sid = cur.lastrowid
    for gid, bs in groups.items():
        for b in bs:
            project.db.execute("INSERT INTO split_assignments(split_id, bottle_id, subset) VALUES(?,?,?)",
                               (sid, b["id"], assign[gid]))
    project.db.commit()
    check_leakage(project, sid)
    return sid


def split_bottles(project: Project, split_id: int, subset: str) -> list[int]:
    return [r[0] for r in project.db.query(
        "SELECT bottle_id FROM split_assignments WHERE split_id=? AND subset=? ORDER BY bottle_id", (split_id, subset))]


def split_id_by_name(project: Project, name: str) -> int:
    r = project.db.one("SELECT id FROM splits WHERE name=?", (name,))
    if r is None:
        raise KeyError(name)
    return r[0]


def check_leakage(project: Project, split_id: int) -> None:
    """Raise if any physical sample appears in more than one subset."""
    rows = project.db.query(
        """SELECT COALESCE(NULLIF(b.physical_sample_id,''), b.bottle_code) AS pid, COUNT(DISTINCT a.subset) AS n
           FROM split_assignments a JOIN bottles b ON a.bottle_id=b.id
           WHERE a.split_id=? AND a.subset != 'excluded' GROUP BY pid HAVING n > 1""", (split_id,))
    if rows:
        raise RuntimeError(f"Split leakage: physical samples in multiple subsets: {[r[0] for r in rows]}")


def splits(project: Project) -> list[dict]:
    out = []
    for r in project.db.query("SELECT * FROM splits ORDER BY id"):
        d = dict(r)
        d["counts"] = {s: n for s, n in project.db.query(
            "SELECT subset, COUNT(*) FROM split_assignments WHERE split_id=? GROUP BY subset", (r["id"],))}
        out.append(d)
    return out
