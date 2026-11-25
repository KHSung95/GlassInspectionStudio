"""Bottle-level split leakage and evaluation semantics under partial labels."""
import pytest

from conftest import make_frames
from gistudio.algorithms.base import Candidate
from gistudio.core.annotations import AnnotationStore
from gistudio.core.splits import check_leakage, make_split, split_bottles
from gistudio.evaluation.metrics import EvalConfig, evaluate_run
from gistudio.inference.engine import save_candidates


def test_split_by_physical_sample_no_leakage(project, tmp_path):
    # B0/B1 are two captures of the same physical bottle P0 -> must stay together
    make_frames(project, tmp_path, n_bottles=8, n_frames=2,
                physical={"B0": "P0", "B1": "P0", "B2": "P2", "B3": "P3", "B4": "P4", "B5": "P5", "B6": "P6", "B7": "P7"})
    sid = make_split(project, "s", (0.5, 0.25, 0.25), seed=3)
    subsets = {}
    for sub in ("train", "val", "test"):
        for b in split_bottles(project, sid, sub):
            subsets[b] = sub
    codes = {b["id"]: b["bottle_code"] for b in project.bottles()}
    by_code = {codes[k]: v for k, v in subsets.items()}
    assert len(by_code) == 8 and by_code["B0"] == by_code["B1"]
    tr = {f["id"] for f in project.all_frames(split_bottles(project, sid, "train"))}
    te = {f["id"] for f in project.all_frames(split_bottles(project, sid, "test"))}
    assert tr and te and not (tr & te)
    # manual corruption is detected
    b1 = [k for k, v in codes.items() if v == "B1"][0]
    other = "test" if by_code["B0"] != "test" else "train"
    project.db.execute("UPDATE split_assignments SET subset=? WHERE split_id=? AND bottle_id=?", (other, sid, b1))
    with pytest.raises(RuntimeError):
        check_leakage(project, sid)


def test_holdout_product_type(project, tmp_path):
    make_frames(project, tmp_path, n_bottles=4, n_frames=1)
    bs = project.bottles()
    project.update_bottle(bs[0]["id"], product_type="X")
    sid = make_split(project, "h", seed=0, holdout_product_types=["X"])
    assert bs[0]["id"] in split_bottles(project, sid, "test")


def test_detections_in_unknown_area_are_not_false_positives(project, tmp_path):
    make_frames(project, tmp_path, n_bottles=1, n_frames=1, shape=(100, 100))
    f = project.all_frames()[0]
    store = AnnotationStore(project)
    store.add(f["id"], "Black Spot", "positive", "point", {"x": 20.5, "y": 20.5})
    store.add(f["id"], "Normal Engraving", "negative", "bbox", {"x0": 60, "y0": 60, "x1": 90, "y1": 90})
    run = project.create_run("classical", "t")
    save_candidates(project, run["id"], f["id"], [
        Candidate(f["id"], "m", "t", "Black Spot", 0.9, "point", {"x": 22.0, "y": 21.0}),   # TP (within 6px)
        Candidate(f["id"], "m", "t", "Black Spot", 0.8, "point", {"x": 70.5, "y": 70.5}),   # FP on verified normal
        Candidate(f["id"], "m", "t", "Black Spot", 0.7, "point", {"x": 40.5, "y": 80.5}),   # unknown area
    ], 5.0)
    r = evaluate_run(project, run["id"], [f["id"]], EvalConfig(tolerance_px=6, operating_threshold=0.5))
    o = r["operating_point"]
    assert (o["tp"], o["fn"], o["fp_verified"], o["unverified"]) == (1, 0, 1, 1)
    assert o["precision_verified"] == 0.5 and o["loc_error_mean"] == pytest.approx((1.5 ** 2 + 0.5 ** 2) ** 0.5)
    assert r["label_completeness"]["status"] == "partial"
    assert r["bottle_level"]["false_reject_rate"] is None and "NOT MEASURABLE" in r["bottle_level"]["false_reject_note"]
    assert any("No verified OK bottles" in w for w in r["warnings"])
    # a too-far detection is not matched by point tolerance (IoU is not used)
    r2 = evaluate_run(project, run["id"], [f["id"]], EvalConfig(tolerance_px=1.0, operating_threshold=0.85))
    assert r2["operating_point"]["tp"] == 0
