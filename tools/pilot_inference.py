"""Pilot: run the configured detectors on captured frames and write a visual report.

    python tools/pilot_inference.py <project> --bottle <code> [--frames N] [--roi x0,y0,x1,y1]
        [--unet <best.pt>] [--winclip] [--out <dir>]

* Classical detectors run on full frames at native resolution.
* WinCLIP runs on the ROI only (CPU cost grows with image area); it is optional.
* A U-Net checkpoint trained on SYNTHETIC data is a domain-mismatched baseline — useful only to check the
  pipeline end-to-end, not for detection quality.
Results are stored as normal runs (candidates in the DB) and summarised in <out>/pilot_report.md with
overlay PNGs. No metric is computed here unless the frames have confirmed labels.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistudio.algorithms.zero_shot.common import configure_model_cache  # noqa: E402
from gistudio.core.annotations import AnnotationStore  # noqa: E402
from gistudio.core.geometry import geometry_center  # noqa: E402
from gistudio.core.image_io import load_mono  # noqa: E402
from gistudio.core.preprocessing import DisplayTransform  # noqa: E402
from gistudio.core.project import Project  # noqa: E402
from gistudio.evaluation.metrics import EvalConfig, evaluate_run  # noqa: E402
from gistudio.inference.engine import candidates, run_on_frames  # noqa: E402

CLASSICAL = [("classical.blackhat", {"threshold": 8.0, "kernel": 11}),
             ("classical.log_blob", {"threshold": 8.0}),
             ("classical.lowfreq_deviation", {"threshold": 4.0}),
             ("classical.bubble_contour", {"threshold": 4.0})]
COLORS = {"classical.blackhat": (0, 0, 255), "classical.log_blob": (0, 165, 255), "classical.lowfreq_deviation": (255, 0, 255),
          "classical.bubble_contour": (0, 255, 255), "zeroshot.winclip": (255, 255, 0), "supervised.unet": (0, 255, 0)}


def overlay(img, bit_depth, cands, path, max_side=2000):
    disp = DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=bit_depth).apply(img)
    s = min(1.0, max_side / max(img.shape))
    rgb = cv2.cvtColor(cv2.resize(disp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    for c in cands:
        x, y = geometry_center(c["geometry_type"], c["geometry"])
        col = COLORS.get(c["source_algorithm"], (255, 255, 255))
        cv2.circle(rgb, (int(x * s), int(y * s)), max(6, int(12 * s * 4)), col, 2)
    ok, enc = cv2.imencode(".png", rgb)
    enc.tofile(str(path))


def main():
    configure_model_cache()
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("--bottle", required=True)
    ap.add_argument("--frames", type=int, default=0, help="limit number of frames (0 = all)")
    ap.add_argument("--roi", default="", help="x0,y0,x1,y1 for WinCLIP (default: centre 960x960)")
    ap.add_argument("--unet", default=""); ap.add_argument("--winclip", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--sweep", action="store_true", help="run label-driven parameter sweeps (needs confirmed labels)")
    ap.add_argument("--sweep-combos", type=int, default=24)
    a = ap.parse_args()
    p = Project(a.project)
    bids = [b["id"] for b in p.bottles() if b["bottle_code"] == a.bottle]
    if not bids:
        raise SystemExit(f"bottle {a.bottle} not found")
    frames = p.all_frames(bids)
    if a.frames:
        frames = frames[: a.frames]
    fids = [f["id"] for f in frames]
    out = Path(a.out or (p.root / "exports" / f"pilot_{a.bottle}_{time.strftime('%Y%m%d-%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=True)
    report = {"bottle": a.bottle, "frames": len(fids), "image": f"{frames[0]['width']}x{frames[0]['height']} {frames[0]['dtype']}",
              "runs": []}
    jobs = list(CLASSICAL)
    if a.unet:
        jobs.append(("supervised.unet", {"checkpoint": a.unet, "tile": 512, "overlap": 64, "threshold_spot": 0.5,
                                         "threshold_area": 0.7}))
    run_ids = {}
    for alg, params in jobs:
        t0 = time.time()
        r = run_on_frames(p, alg, fids, params, save_maps=False, notes=f"pilot {a.bottle}")
        run_ids[alg] = r["id"]
        n = sum(len(candidates(p, r["id"], f)) for f in fids)
        report["runs"].append({"algorithm": alg, "run_id": r["id"], "params": params, "candidates": n,
                               "per_frame": round(n / max(1, len(fids)), 2), "ms_frame": round(r["summary"]["ms_mean"], 1),
                               "wall_s": round(time.time() - t0, 1)})
        print(report["runs"][-1], flush=True)
    if a.winclip:
        from gistudio.algorithms.base import create
        h, w = frames[0]["height"], frames[0]["width"]
        if a.roi:
            x0, y0, x1, y1 = (int(v) for v in a.roi.split(","))
        else:
            x0, y0 = max(0, w // 2 - 480), max(0, h // 2 - 480); x1, y1 = min(w, x0 + 960), min(h, y0 + 960)
        alg = create("zeroshot.winclip", tile=240, overlap=0, batch=8, threshold=0.0)
        f = frames[0]
        img = load_mono(f["path"])[y0:y1, x0:x1]
        t0 = time.time()
        res = alg.run(img, {"bit_depth": f.get("bit_depth")})
        m = res.score_maps["winclip_score"]
        hm = cv2.applyColorMap(np.clip((m - m.min()) / max(1e-9, m.max() - m.min()) * 255, 0, 255).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        ok, enc = cv2.imencode(".png", hm); enc.tofile(str(out / "winclip_roi_scoremap.png"))
        disp = DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=f.get("bit_depth")).apply(img)
        ok, enc = cv2.imencode(".png", disp); enc.tofile(str(out / "winclip_roi_image.png"))
        report["winclip"] = {"frame_id": f["id"], "roi": [x0, y0, x1, y1], "seconds": round(time.time() - t0, 1),
                             "score_min": float(m.min()), "score_p50": float(np.median(m)), "score_p99": float(np.percentile(m, 99)),
                             "score_max": float(m.max()), "warnings": res.warnings}
        print(report["winclip"], flush=True)
    # overlays for up to 6 frames
    for f in frames[:6]:
        cs = [c for rid in run_ids.values() for c in candidates(p, rid, f["id"])]
        overlay(load_mono(f["path"]), f.get("bit_depth"), cs, out / f"overlay_f{f['id']}_a{f['angle_deg']}.png")
    # metrics only if confirmed labels exist
    store = AnnotationStore(p)
    labelled = [f for f in fids if any(x["confirmed"] and x["label_state"] == "positive" for x in store.for_frame(f))]
    if labelled:
        report["evaluation_on_labelled_frames"] = {}
        for alg, rid in run_ids.items():
            e = evaluate_run(p, rid, labelled, EvalConfig())
            report["evaluation_on_labelled_frames"][alg] = {"n_gt": e["n_gt"], "recall@lowest": e["operating_point"]["recall"],
                                                             "label_completeness": e["label_completeness"]["status"]}
    else:
        report["evaluation_on_labelled_frames"] = "no confirmed positive labels yet — label a few defects first"
    # label-driven parameter sweep for the spot detectors (classical only)
    if labelled and a.sweep:
        from gistudio.evaluation.param_sweep import SweepConfig, labelled_frames, run_sweep, save_presets
        lf = labelled_frames(p, bids)
        report["sweep"] = {}
        for alg in ("classical.blackhat", "classical.log_blob", "classical.dog"):
            try:
                r = run_sweep(p, SweepConfig(alg, lf, max_combos=a.sweep_combos))
                save_presets(p, r, prefix=f"pilot_{a.bottle}")
                report["sweep"][alg] = {"n_gt": r["n_gt"], "baseline": r["baseline_defaults"]["tune"],
                                        "recommendations": [{"objective": x["objective"], "params": x["params"],
                                                             "tune": x["tune"]} for x in r["recommendations"]],
                                        "warnings": r["warnings"]}
                print(alg, "sweep done", flush=True)
            except Exception as e:
                report["sweep"][alg] = {"error": repr(e)}
    (out / "pilot_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = [f"# Pilot report — bottle {a.bottle}", "", f"* frames: {report['frames']} ({report['image']})", "",
             "| algorithm | run | candidates | per frame | ms/frame |", "|---|---|---|---|---|"]
    lines += [f"| {r['algorithm']} | {r['run_id']} | {r['candidates']} | {r['per_frame']} | {r['ms_frame']} |" for r in report["runs"]]
    if "winclip" in report:
        wcl = report["winclip"]
        lines += ["", f"WinCLIP ROI {wcl['roi']}: {wcl['seconds']} s, score min/p50/p99/max = "
                      f"{wcl['score_min']:.3f}/{wcl['score_p50']:.3f}/{wcl['score_p99']:.3f}/{wcl['score_max']:.3f}"]
    for alg, s in report.get("sweep", {}).items():
        if "error" in s:
            lines += ["", f"## Sweep {alg}: {s['error']}"]; continue
        b = s["baseline"]
        lines += ["", f"## Parameter sweep — {alg} (GT={s['n_gt']})", "",
                  "| preset | threshold | recall | precision(verified) | FP verified/frame | unverified/frame | params |",
                  "|---|---|---|---|---|---|---|",
                  f"| defaults | {b['threshold']} | {b['recall']} | {b['precision_verified']} | {b['fp_per_frame']} | "
                  f"{b['unverified_per_frame']} | - |"]
        for x in s["recommendations"]:
            m = x["tune"]
            pr = {k: v for k, v in x["params"].items() if k not in ("threshold", "max_candidates", "max_area")}
            lines.append(f"| {x['objective']} | {m['threshold']} | {m['recall']} | {m['precision_verified']} | "
                         f"{m['fp_per_frame']} | {m['unverified_per_frame']} | `{json.dumps(pr)}` |")
        lines += [f"* {w}" for w in s["warnings"]]
    lines += ["", f"Evaluation: {report['evaluation_on_labelled_frames']}", "",
              "Candidates are unverified. A synthetic-trained U-Net is a domain-mismatched baseline."]
    (out / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
