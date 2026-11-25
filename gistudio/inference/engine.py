"""Batch inference over frames: runs any registered algorithm, stores candidates + score maps + timing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..algorithms.base import Candidate, InspectionAlgorithm, create
from ..core.image_io import load_mono
from ..core.project import Project


def save_candidates(project: Project, run_id: int, frame_id: int, cands: list[Candidate], inference_ms: float) -> list[int]:
    ids = []
    for c in cands:
        cur = project.db.execute(
            """INSERT INTO candidates(run_id, frame_id, model_id, source_algorithm, class_name, score, geometry_type,
               geometry, area_px, inference_ms, review_status, features) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, frame_id, c.model_id, c.source_algorithm, c.class_name, float(c.score), c.geometry_type,
             json.dumps(c.geometry, default=float), float(c.area_px), inference_ms, "pending",
             json.dumps(c.features, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))
        ids.append(cur.lastrowid)
    project.db.commit()
    return ids


def run_on_frames(project: Project, algorithm: InspectionAlgorithm | str, frame_ids: list[int], params: dict | None = None,
                  save_maps: bool = True, progress=None, stop_flag=None, notes: str = "") -> dict:
    alg = create(algorithm, **(params or {})) if isinstance(algorithm, str) else algorithm
    kind = {"classical": "classical", "zero_shot": "zeroshot"}.get(alg.family, "infer")
    run = project.create_run(kind, alg.algorithm_id, params=alg.params, model_info=alg.info(),
                             score_semantics=alg.score_semantics, notes=notes)
    out_dir: Path = run["output_dir"]
    timings, warnings, n_c = [], set(), 0
    try:
        for i, fid in enumerate(frame_ids):
            if stop_flag is not None and stop_flag():
                project.finish_run(run["id"], "stopped")
                return {**run, "status": "stopped"}
            f = project.frame(fid)
            img = load_mono(f["path"])
            t0 = time.perf_counter()
            res = alg.run(img, {"bit_depth": f.get("bit_depth"), "frame_id": fid, "angle_deg": f.get("angle_deg")})
            ms = (time.perf_counter() - t0) * 1000
            timings.append(ms)
            warnings.update(res.warnings)
            for c in res.candidates:
                c.frame_id = fid
            save_candidates(project, run["id"], fid, res.candidates, ms)
            n_c += len(res.candidates)
            if save_maps and res.score_maps:
                np.savez_compressed(out_dir / f"maps_{fid}.npz",
                                    **{k: v.astype(np.float16) for k, v in res.score_maps.items()})
            if progress:
                progress(i + 1, len(frame_ids), f"frame {fid}: {len(res.candidates)} candidates, {ms:.0f} ms")
        summary = {"frames": len(frame_ids), "candidates": n_c, "ms_mean": float(np.mean(timings)) if timings else 0,
                   "ms_max": float(np.max(timings)) if timings else 0, "warnings": sorted(warnings),
                   "model_info": res.model_info if frame_ids else {}, "preprocess": res.preprocess if frame_ids else {}}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        project.db.execute("UPDATE runs SET preprocess=?, model_info=? WHERE id=?",
                           (json.dumps(summary["preprocess"], default=str),
                            json.dumps({**alg.info(), **summary["model_info"]}, default=str), run["id"]))
        project.finish_run(run["id"], "done")
        return {**run, "status": "done", "summary": summary}
    except Exception as e:
        project.finish_run(run["id"], "failed", notes=repr(e))
        raise


def load_maps(project: Project, run_id: int, frame_id: int) -> dict[str, np.ndarray]:
    r = project.run(run_id)
    f = project.root / r["output_dir"] / f"maps_{frame_id}.npz"
    if not f.exists():
        return {}
    d = np.load(f)
    return {k: d[k].astype(np.float32) for k in d.files}


def candidates(project: Project, run_id: int | None = None, frame_id: int | None = None,
               review_status: str | None = None, min_score: float | None = None) -> list[dict]:
    q, p = "SELECT * FROM candidates WHERE 1=1", []
    for col, val in (("run_id", run_id), ("frame_id", frame_id), ("review_status", review_status)):
        if val is not None:
            q += f" AND {col}=?"; p.append(val)
    if min_score is not None:
        q += " AND score>=?"; p.append(min_score)
    out = []
    for r in project.db.query(q + " ORDER BY score DESC", p):
        d = dict(r)
        d["geometry"] = json.loads(d["geometry"])
        d["features"] = json.loads(d["features"] or "{}")
        out.append(d)
    return out
