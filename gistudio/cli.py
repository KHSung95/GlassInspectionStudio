"""Command line interface. Every heavy operation the GUI offers is available here.

    python -m gistudio.cli <command> ...   (see `--help`)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _frames(project, spec: str) -> list[int]:
    """'all' | 'split:<name>:<subset>' | 'bottle:<code>' | comma separated frame ids."""
    from .core.splits import split_bottles, split_id_by_name
    if spec == "all":
        return [f["id"] for f in project.all_frames()]
    if spec.startswith("split:"):
        _, name, subset = spec.split(":")
        return [f["id"] for f in project.all_frames(split_bottles(project, split_id_by_name(project, name), subset))]
    if spec.startswith("bottle:"):
        code = spec.split(":", 1)[1]
        ids = [b["id"] for b in project.bottles() if b["bottle_code"] == code]
        return [f["id"] for f in project.all_frames(ids)]
    return [int(x) for x in spec.split(",") if x]


def _progress(i, n, msg):
    print(f"[{i}/{n}] {msg}", flush=True)


def main(argv=None) -> int:
    from .algorithms.zero_shot.common import configure_model_cache
    configure_model_cache()
    ap = argparse.ArgumentParser(prog="gistudio", description="Glass Inspection Studio CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="Show Python / PyTorch / GPU environment")
    sub.add_parser("algorithms", help="List registered algorithms")

    p = sub.add_parser("create", help="Create a new project directory"); p.add_argument("project")

    p = sub.add_parser("import", help="Import an image folder")
    p.add_argument("project"); p.add_argument("folder")
    p.add_argument("--layout", default="bottle/session", choices=["bottle/session", "bottle", "flat"])
    p.add_argument("--dataset", default="default"); p.add_argument("--bottle", default="")
    p.add_argument("--angle-step", type=float); p.add_argument("--angle-csv"); p.add_argument("--angle-regex")

    p = sub.add_parser("synth", help="Generate synthetic verification data")
    p.add_argument("out"); p.add_argument("--bottles", type=int, default=4); p.add_argument("--angles", type=int, default=36)
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=512)
    p.add_argument("--sessions", type=int, default=1); p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("demo", help="Synthetic data -> new project with partial labels and a bottle-level split")
    p.add_argument("project"); p.add_argument("--bottles", type=int, default=6); p.add_argument("--angles", type=int, default=36)
    p.add_argument("--width", type=int, default=640); p.add_argument("--height", type=int, default=512)
    p.add_argument("--sessions", type=int, default=1)

    p = sub.add_parser("split", help="Create a bottle-level split")
    p.add_argument("project"); p.add_argument("--name", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", default="0.6,0.2,0.2"); p.add_argument("--holdout-product", action="append", default=[])

    p = sub.add_parser("run", help="Run an algorithm over frames and store candidates")
    p.add_argument("project"); p.add_argument("--algorithm", required=True)
    p.add_argument("--params", default="{}", help="JSON dict of parameters"); p.add_argument("--frames", default="all")
    p.add_argument("--no-maps", action="store_true")

    p = sub.add_parser("train-unet", help="Train the high-resolution U-Net (masked loss)")
    p.add_argument("project"); p.add_argument("--split", required=True); p.add_argument("--config", default="{}",
                                                                                       help="JSON overrides or path")
    p.add_argument("--run-dir", default=""); p.add_argument("--resume", default="")

    p = sub.add_parser("train-patch", help="Train the multi-scale Patch CNN")
    p.add_argument("project"); p.add_argument("--split", required=True); p.add_argument("--config", default="{}")
    p.add_argument("--run-dir", default="")

    p = sub.add_parser("eval", help="Evaluate one or more runs on the same frames")
    p.add_argument("project"); p.add_argument("--runs", required=True, help="comma separated run ids")
    p.add_argument("--frames", default="all"); p.add_argument("--tol", type=float, default=6.0)
    p.add_argument("--threshold", type=float); p.add_argument("--report", default="")

    p = sub.add_parser("aggregate", help="Multi-view bottle-level aggregation")
    p.add_argument("project"); p.add_argument("--run", type=int, required=True); p.add_argument("--bottle", required=True)
    p.add_argument("--threshold", type=float, default=0.5)

    p = sub.add_parser("angular", help="Angular visibility analysis for all defects of a bottle")
    p.add_argument("project"); p.add_argument("--bottle", required=True); p.add_argument("--run", type=int)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--steps", default="1,2,5,10,15,30,90")
    p.add_argument("--out", default="")

    p = sub.add_parser("cameras", help="List GenICam cameras via a GenTL producer (read-only)")
    p.add_argument("--cti", default=None)

    p = sub.add_parser("capture", help="Capture frames into a project (manual rotation; Enter between shots)")
    p.add_argument("project"); p.add_argument("--bottle", required=True); p.add_argument("--session", default="")
    p.add_argument("--product-type", default=""); p.add_argument("--n", type=int, default=1)
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--step", type=float, default=0.0,
                                                                        help="angle step per shot; 0 = no angle")
    p.add_argument("--avg", type=int, default=1); p.add_argument("--interactive", action="store_true")
    p.add_argument("--mock", action="store_true", help="use the synthetic mock camera"); p.add_argument("--cti", default=None)
    p.add_argument("--exposure", type=float, help="WRITE ExposureTime (µs) before capture (explicit)")
    p.add_argument("--gain", type=float, help="WRITE Gain (dB) before capture (explicit)")

    p = sub.add_parser("sweep", help="Label-driven parameter sweep of a classical detector -> recommended presets")
    p.add_argument("project"); p.add_argument("--algorithm", required=True)
    p.add_argument("--bottle", default="", help="restrict to labelled frames of this bottle code")
    p.add_argument("--max-combos", type=int, default=60); p.add_argument("--tol", type=float, default=6.0)
    p.add_argument("--margin", type=int, default=160); p.add_argument("--holdout", type=float, default=0.0)
    p.add_argument("--recall-target", type=float, default=0.9); p.add_argument("--fp-budget", type=float, default=0.2)
    p.add_argument("--save", action="store_true", help="save recommendations as project presets")
    p.add_argument("--classes", default="", help="GT classes for this detector, comma separated "
                                                 "(default: the detector's own defect type)")

    p = sub.add_parser("console", help="Local web camera console (live feed + capture) on 127.0.0.1")
    p.add_argument("project"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mock", action="store_true"); p.add_argument("--cti", default=None)
    p.add_argument("--packet-size", type=int, help="WRITE GevSCPSPacketSize at start (needs jumbo frames on the NIC)")
    p.add_argument("--fps", type=float, help="WRITE AcquisitionFrameRate at start (bandwidth headroom)")

    p = sub.add_parser("label", help="Browser draft-labelling page on 127.0.0.1 (no camera)")
    p.add_argument("project"); p.add_argument("--port", type=int, default=8770)
    p.add_argument("--author", default="operator")

    p = sub.add_parser("studio", help="Unified web studio (project / capture / labelling / analysis / report)")
    p.add_argument("--port", type=int, default=8800); p.add_argument("--project", default=None)
    p.add_argument("--open-browser", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "studio":
        from .app.web_studio import serve as serve_studio
        serve_studio(a.port, a.project, a.open_browser); return 0
    if a.cmd == "label":
        from .app.web_label import serve as serve_label
        serve_label(a.project, a.port, a.author); return 0
    if a.cmd == "console":
        from .acquisition.web_console import serve
        startup = {}
        if a.packet_size:
            startup["GevSCPSPacketSize"] = a.packet_size
        if a.fps:
            startup["AcquisitionFrameRate"] = a.fps
        serve(a.project, a.port, a.mock, a.cti, startup); return 0
    if a.cmd == "cameras":
        from .acquisition.genicam_camera import DEFAULT_CTI, list_devices
        print(json.dumps(list_devices(a.cti or DEFAULT_CTI), indent=1, default=str)); return 0

    if a.cmd == "env":
        from .core.device import detect
        print(json.dumps(detect(), indent=2)); return 0
    if a.cmd == "algorithms":
        from .algorithms.base import registry
        for k, v in registry().items():
            print(f"{k:38s} [{v.family}] {v.display_name}")
        return 0
    if a.cmd == "synth":
        from .synthetic import generate_dataset
        print(generate_dataset(a.out, a.bottles, a.angles, a.width, a.height, a.seed, a.sessions)); return 0

    from .core.project import Project
    if a.cmd == "create":
        Project.create(a.project); print(f"created {a.project}"); return 0
    if a.cmd == "demo":
        from .core.dataset_import import ImportOptions, import_folder
        from .core.splits import make_split
        from .synthetic import generate_dataset, import_synthetic_labels
        root = Path(a.project)
        prj = Project.create(root)
        data = generate_dataset(root / "synthetic_source", a.bottles, a.angles, a.width, a.height, 0, a.sessions)
        rep = import_folder(prj, data, ImportOptions(layout="bottle/session", dataset_name="synthetic"), _progress)
        print(rep.summary())
        print(import_synthetic_labels(prj, data))
        sid = make_split(prj, "bottle-split", (0.5, 0.25, 0.25), seed=0)
        print(f"split id {sid}"); return 0

    prj = Project(a.project)
    if a.cmd == "import":
        from .core.dataset_import import ImportOptions, import_folder
        o = ImportOptions(layout=a.layout, dataset_name=a.dataset, bottle_code=a.bottle, angle_step=a.angle_step,
                          angle_csv=a.angle_csv)
        if a.angle_regex:
            o.angle_regex = a.angle_regex
        rep = import_folder(prj, a.folder, o, _progress)
        print(rep.summary())
        for d in rep.duplicates[:20]:
            print("duplicate:", d)
        for e in rep.errors[:20]:
            print("error:", e)
        return 0 if not rep.errors else 1
    if a.cmd == "capture":
        from .acquisition.capture_session import CaptureSessionWriter, CaptureTarget
        if a.mock:
            from .acquisition.mock import MockCamera, MockRotationStage
            stage = MockRotationStage(); cam = MockCamera(stage, 1024, 768)
        else:
            from .acquisition.genicam_camera import DEFAULT_CTI, GenICamCamera
            cam = GenICamCamera(a.cti or DEFAULT_CTI)
        cam.open()
        try:
            if a.exposure is not None:
                cam.set_exposure(a.exposure)
            if a.gain is not None:
                cam.set_gain(a.gain)
            info = cam.info()
            print({k: v for k, v in info.items() if k != "features"})
            w = CaptureSessionWriter(prj, CaptureTarget(a.bottle, a.session, a.product_type), info)
            for k in range(a.n):
                ang = a.start + k * a.step if a.step else None
                if a.interactive:
                    input(f"[{k + 1}/{a.n}] set the bottle to {ang if ang is not None else '-'} deg, then press Enter ")
                if a.mock:
                    stage.move_to(ang or 0.0)  # mock only: simulated stage
                img = cam.grab_averaged(a.avg) if hasattr(cam, "grab_averaged") else cam.grab()
                fid = w.add(img, ang, info.get("exposure_us"), info.get("gain_db"))
                print(f"frame {fid}: {img.shape} {img.dtype} min={img.min()} max={img.max()} angle={ang}")
            print(f"session {w.session_code} -> {w.dir}")
        finally:
            cam.close()
        return 0
    if a.cmd == "sweep":
        from .evaluation.param_sweep import SweepConfig, labelled_frames, run_sweep, save_presets
        bids = [b["id"] for b in prj.bottles() if b["bottle_code"] == a.bottle] if a.bottle else None
        fids = labelled_frames(prj, bids)
        gtc = [c.strip() for c in a.classes.split(",") if c.strip()] or None
        r = run_sweep(prj, SweepConfig(a.algorithm, fids, None, gtc, a.tol, a.margin, a.max_combos, 0,
                                       a.recall_target, a.fp_budget, a.holdout), _progress)
        print(json.dumps({k: v for k, v in r.items() if k != "config"}, indent=1, default=str))
        if a.save:
            for path in save_presets(prj, r):
                print("preset:", path)
        return 0
    if a.cmd == "split":
        from .core.splits import make_split
        print(make_split(prj, a.name, tuple(float(x) for x in a.ratios.split(",")), a.seed,
                         holdout_product_types=a.holdout_product)); return 0
    if a.cmd == "run":
        from .inference.engine import run_on_frames
        params = json.loads(Path(a.params).read_text() if Path(a.params).suffix == ".json" else a.params)
        r = run_on_frames(prj, a.algorithm, _frames(prj, a.frames), params, not a.no_maps, _progress)
        print(json.dumps({"run_id": r["id"], "run_uid": r["run_uid"], **r.get("summary", {})}, indent=2, default=str))
        return 0
    if a.cmd == "train-unet":
        from .training.trainers import UNetTrainConfig, train_unet
        over = json.loads(Path(a.config).read_text() if a.config.endswith(".json") else a.config)
        cfg = UNetTrainConfig.from_dict({**over, "project": str(prj.root), "split": a.split, "run_dir": a.run_dir,
                                         "resume": a.resume})
        print(train_unet(cfg)); return 0
    if a.cmd == "train-patch":
        from .training.trainers import PatchTrainConfig, train_patch_cnn
        over = json.loads(Path(a.config).read_text() if a.config.endswith(".json") else a.config)
        print(train_patch_cnn(PatchTrainConfig.from_dict({**over, "project": str(prj.root), "split": a.split,
                                                          "run_dir": a.run_dir}))); return 0
    if a.cmd == "eval":
        from .evaluation.metrics import EvalConfig, compare_runs, write_report
        res = compare_runs(prj, [int(x) for x in a.runs.split(",")], _frames(prj, a.frames),
                           EvalConfig(tolerance_px=a.tol, operating_threshold=a.threshold))
        for r in res:
            r = {k: v for k, v in r.items() if k not in ("curve",)}
            print(json.dumps(r, indent=1, default=str))
        if a.report:
            write_report(res, a.report); print(f"report -> {a.report}")
        return 0
    if a.cmd == "aggregate":
        from .inference.multiview import AggregationConfig, aggregate_bottle
        b = [x for x in prj.bottles() if x["bottle_code"] == a.bottle][0]
        print(json.dumps(aggregate_bottle(prj, a.run, b["id"], AggregationConfig(score_threshold=a.threshold)),
                         indent=1, default=str)); return 0
    if a.cmd == "angular":
        from .core.annotations import AnnotationStore
        from .evaluation.angular import defect_profile, sampling_experiment, visible_intervals
        b = [x for x in prj.bottles() if x["bottle_code"] == a.bottle][0]
        profs = [defect_profile(prj, d["id"], a.run) for d in AnnotationStore(prj).defects(b["id"])]
        key = "score" if a.run else "visibility"
        out = {"defects": [{"uid": p["defect_uid"], "best_view_by_score": p["best_view_by_score"],
                            "best_view_by_contrast": p["best_view_by_contrast"], "coverage": p["coverage"],
                            "intervals": visible_intervals(p, key, a.threshold)} for p in profs],
               "sampling": sampling_experiment(profs, [float(s) for s in a.steps.split(",")], key=key, thr=a.threshold)}
        txt = json.dumps(out, indent=1, default=str)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
        print(txt); return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
