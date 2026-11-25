"""Per-defect-class classical sweep on a labelled project (each class has its own detectors and criteria).

    python tools/per_class_sweep.py <project> <out_dir> [--combos 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.core.project import Project  # noqa: E402
from gistudio.evaluation.param_sweep import SweepConfig, labelled_frames, run_sweep, save_presets  # noqa: E402

PLAN = {
    # class group -> (gt classes, point tolerance px, [(detector, search space or None)])
    "black_spot": (["Black Spot"], 6.0, [("classical.log_blob", None), ("classical.dog", None),
                                         ("classical.blackhat", None)]),
    "bubble": (["Bubble", "Open Bubble", "Closed Bubble"], 25.0, [
        ("classical.log_blob", {"sigma_min": [3.0, 6.0], "sigma_max": [16.0, 32.0], "n_scales": [4],
                                "max_area": [20000, 200000]}),
        ("classical.dog", {"sigma_small": [3.0, 6.0], "sigma_large": [15.0, 30.0], "max_area": [20000, 200000]}),
        ("classical.bubble_contour", {"sigma_min": [3.0, 6.0], "sigma_max": [16.0, 32.0],
                                      "min_circularity": [0.2, 0.4]})]),
    "intensity_tint": (["Intensity / Transparency Defect", "Faint Tint"], 15.0, [
        ("classical.lowfreq_deviation", {"scales": ["8,16,32", "16,32,64"], "context_factor": [3.0, 5.0],
                                         "min_area": [200, 1000]}),
        ("classical.local_intensity_deviation", {"smooth_sigma": [4.0, 8.0], "bg_kernel": [151, 301],
                                                 "min_area": [200, 1000]})]),
    "linear_void": (["Linear / Broken Void", "Other Defect"], 15.0, [
        ("classical.ridge_void", None),
        ("classical.gradient_texture", {"window": [9, 15], "context": [151, 301], "min_area": [30, 100]})]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("out"); ap.add_argument("--combos", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    p = Project(a.project)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fids = labelled_frames(p)
    summary = {}
    for group, (classes, tol, dets) in PLAN.items():
        if a.only and group not in a.only.split(","):
            continue
        for alg, space in dets:
            t0 = time.time()
            try:
                r = run_sweep(p, SweepConfig(alg, fids, space, classes, tol, 200, a.combos, 0, 0.9, 0.2))
            except Exception as e:
                summary.setdefault(group, {})[alg] = {"error": repr(e)}
                print(group, alg, "ERROR", repr(e), flush=True)
                continue
            save_presets(p, r, prefix=f"{group}")
            (out / f"sweep_{group}_{alg.split('.')[-1]}.json").write_text(json.dumps(r, indent=1, default=str),
                                                                          encoding="utf-8")
            b = r["baseline_defaults"]["tune"]
            summary.setdefault(group, {})[alg] = {
                "n_gt": r["n_gt"], "seconds": round(time.time() - t0, 1), "baseline": b,
                "recommendations": [{"objective": x["objective"], "params": x["params"], "tune": x["tune"]}
                                    for x in r["recommendations"]], "warnings": r["warnings"]}
            print(f"{group:15s} {alg:36s} GT {r['n_gt']:2d}  base recall {b['recall']} fa/MP {b['false_alarms_per_mp']}",
                  flush=True)
            for x in r["recommendations"]:
                m = x["tune"]
                print(f"    {x['objective']:17s} thr {m['threshold']:<6} recall {m['recall']:<6} "
                      f"FPver {m['fp_verified']:<3} fa/MP {m['false_alarms_per_mp']}", flush=True)
    (out / "per_class_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("done", out)


if __name__ == "__main__":
    main()
