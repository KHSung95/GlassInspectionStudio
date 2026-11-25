"""End-to-end smoke test on synthetic data: import -> labels -> split -> classical -> U-Net train -> inference ->
evaluation -> review -> retraining data. SOFTWARE verification only (no claim about real detection performance)."""
import json
import math

import numpy as np
import pytest

from gistudio.algorithms.base import create, registry
from gistudio.core.annotations import AnnotationStore
from gistudio.core.dataset_import import ImportOptions, import_folder
from gistudio.core.image_io import read_image
from gistudio.core.project import Project
from gistudio.core.splits import make_split, split_bottles
from gistudio.evaluation.angular import defect_profile, sampling_experiment
from gistudio.evaluation.metrics import EvalConfig, evaluate_run
from gistudio.inference.candidate_management import apply_group_review, review_queue
from gistudio.inference.engine import candidates, run_on_frames
from gistudio.inference.multiview import AggregationConfig, aggregate_bottle
from gistudio.synthetic import generate_dataset, import_synthetic_labels


@pytest.fixture(scope="module")
def synth_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data = generate_dataset(root / "data", n_bottles=4, n_angles=12, width=192, height=160, seed=1, label_every=2)
    p = Project.create(root / "proj")
    rep = import_folder(p, data, ImportOptions(dataset_name="synthetic"))
    assert rep.imported == 48 and not rep.errors
    n = import_synthetic_labels(p, data)
    assert n["positives"] > 0
    make_split(p, "s", (0.5, 0.25, 0.25), seed=0)
    return p, data


def test_synthetic_data_properties(synth_project):
    p, data = synth_project
    gt = json.loads((data / "B001" / "S001" / "ground_truth.json").read_text())
    kinds = {d["kind"] for d in gt["defects"]}
    assert kinds == {"black_spot", "blotch", "bubble"}
    assert any(not d["labeled"] for d in gt["defects"])                     # unlabelled defects exist
    assert any(d["vis_center"] is not None for d in gt["defects"])          # angle-dependent visibility
    assert any(d["on_structure"] for d in gt["defects"])                    # defect on normal structure
    assert min(d["size"] for d in gt["defects"] if d["kind"] == "black_spot") <= 1.0
    xs = [next((d["x"] for d in fr["defects"] if d["uid"] == "B001-D1"), None) for fr in gt["frames"]]
    xs = [x for x in xs if x is not None]
    assert len(set(round(x) for x in xs)) > 1                               # defect moves with rotation
    img = read_image(data / "B004" / "S001" / "frame_0000.tif")
    assert img.dtype == np.uint16 and img.max() > 4095                     # one bottle with full 16-bit data


def test_classical_detects_synthetic_spots(synth_project):
    p, _ = synth_project
    fr = [f["id"] for f in p.all_frames()]
    r = run_on_frames(p, "classical.log_blob", fr, {"threshold": 5.0})
    e = evaluate_run(p, r["id"], fr, EvalConfig(operating_threshold=5.0))
    assert e["operating_point"]["recall"] > 0.5
    assert e["timing"]["frame_ms_mean"] > 0


def test_all_classical_algorithms_run(synth_project):
    p, _ = synth_project
    img = read_image(p.all_frames()[0]["path"])
    for k in registry("classical"):
        res = create(k).run(img, {})
        assert res.score_maps and all(m.shape == img.shape for m in res.score_maps.values()), k


def test_unet_train_infer_eval_review_cycle(synth_project, tmp_path):
    from gistudio.training.trainers import UNetTrainConfig, train_unet
    p, _ = synth_project
    cfg = UNetTrainConfig(project=str(p.root), split="s", epochs=2, batch_size=2, tile=64, samples_per_epoch=8,
                          val_samples=4, base_channels=8, depth=3, run_dir=str(tmp_path / "unet"), seed=0)
    ck = train_unet(cfg)
    assert ck.exists() and (tmp_path / "unet" / "last.pt").exists()
    lines = [json.loads(l) for l in (tmp_path / "unet" / "metrics.jsonl").read_text().splitlines()]
    ep = [l for l in lines if l["event"] == "epoch"]
    assert len(ep) == 2 and all(math.isfinite(l["train_loss"]) for l in ep)
    # resume continues from the saved epoch
    cfg2 = UNetTrainConfig(**{**cfg.__dict__, "epochs": 3, "resume": str(tmp_path / "unet" / "last.pt")})
    train_unet(cfg2)
    ep = [json.loads(l) for l in (tmp_path / "unet" / "metrics.jsonl").read_text().splitlines()]
    assert [l["epoch"] for l in ep if l["event"] == "epoch"] == [0, 1, 2]
    sid = p.db.one("SELECT id FROM splits WHERE name='s'")[0]
    test_frames = [f["id"] for f in p.all_frames(split_bottles(p, sid, "test"))]
    r = run_on_frames(p, "supervised.unet", test_frames, {"checkpoint": str(ck), "tile": 96, "overlap": 16,
                                                           "threshold_spot": 0.05, "threshold_area": 0.5})
    assert r["status"] == "done"
    e = evaluate_run(p, r["id"], test_frames)
    assert e["n_gt"] > 0 and e["operating_point"]["recall"] is not None
    # review: accept the top candidate (as a group), reject the next -> new GT + hard negative
    q = review_queue(p, r["id"], "high_score", threshold=0.05, limit=5)
    assert q, "inference produced no candidates to review"
    store = AnnotationStore(p)
    before = len(store.search(source="ai_accepted"))
    apply_group_review(p, q[0]["id"], "accept")
    assert len(store.search(source="ai_accepted")) > before
    if len(q) > 1 and p.db.one("SELECT review_status FROM candidates WHERE id=?", (q[1]["id"],))[0] == "pending":
        apply_group_review(p, q[1]["id"], "reject")
        assert store.search(source="ai_rejected")


def test_training_is_reproducible_with_seed(synth_project, tmp_path):
    from gistudio.training.trainers import UNetTrainConfig, train_unet
    p, _ = synth_project
    losses = []
    for k in range(2):
        d = tmp_path / f"r{k}"
        train_unet(UNetTrainConfig(project=str(p.root), split="s", epochs=1, batch_size=2, tile=64, samples_per_epoch=6,
                                   val_samples=2, base_channels=8, depth=2, run_dir=str(d), seed=123))
        ep = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
        losses.append([l["train_loss"] for l in ep if l["event"] == "epoch"] + [l["val_loss"] for l in ep if l["event"] == "epoch"])
        assert json.loads((d / "config.json").read_text())["seed"] == 123
    assert losses[0] == losses[1]


def test_multiview_and_angular_analysis(synth_project):
    p, _ = synth_project
    fr = [f["id"] for f in p.all_frames()]
    r = run_on_frames(p, "classical.blackhat", fr, {"threshold": 6.0})
    b = p.bottles()[0]
    agg = aggregate_bottle(p, r["id"], b["id"], AggregationConfig(score_threshold=6.0))
    assert agg["groups"] and agg["bottle_max_score"] >= 6.0
    assert all(g["n_views"] >= 1 for g in agg["groups"])
    store = AnnotationStore(p)
    profs = [defect_profile(p, d["id"], r["id"]) for d in store.defects(b["id"])]
    assert profs and any(pr["coverage"]["annotated"] > 0 for pr in profs)
    s = sampling_experiment(profs, steps=(30, 90), n_starts=4, key="score", thr=6.0)
    assert set(s) == {"30", "90"}


def test_patch_cnn_train_and_cascade(synth_project, tmp_path):
    from gistudio.training.trainers import PatchTrainConfig, train_patch_cnn
    p, _ = synth_project
    ck = train_patch_cnn(PatchTrainConfig(project=str(p.root), split="s", epochs=1, batch_size=8,
                                          crop_sizes=[16, 48], branch_px=16, width=8, run_dir=str(tmp_path / "pc")))
    assert ck.exists()
    img = read_image(p.all_frames()[0]["path"])
    res = create("supervised.patch_cnn", checkpoint=str(ck), threshold=0.0).run(img, {})
    assert res.model_info["frontend_candidates"] >= len(res.candidates)
