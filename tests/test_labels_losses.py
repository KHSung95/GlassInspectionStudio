"""Partial-label semantics: Unknown never becomes normal, masked losses, empty batches, persistence,
AI candidates kept out of ground truth."""
import json

import numpy as np
import pytest
import torch

from conftest import make_frames
from gistudio.algorithms.base import Candidate
from gistudio.core.annotations import AnnotationStore
from gistudio.core.label_targets import TargetPolicy, build_targets
from gistudio.core.project import Project
from gistudio.inference.engine import save_candidates
from gistudio.training.losses import (combined_loss, masked_bce, masked_dice, masked_focal, masked_heatmap_focal,
                                      masked_mse)


@pytest.fixture
def frame(project, tmp_path):
    make_frames(project, tmp_path, n_bottles=1, n_frames=2, shape=(64, 80))
    return project.all_frames()[0]


def test_unlabelled_pixels_are_unknown_not_normal(project, frame):
    store = AnnotationStore(project)
    shape = (frame["height"], frame["width"])
    t, v, _ = build_targets(store, frame["id"], shape)
    assert not v.any() and not t.any()              # NG bottle frame without labels: nothing known
    store.add(frame["id"], "Black Spot", "positive", "point", {"x": 20.5, "y": 30.5, "r": 1})
    t, v, _ = build_targets(store, frame["id"], shape, TargetPolicy(point_valid_radius=5))
    ch = 0
    assert t[ch, 30, 20] == 1.0 and v[ch, 30, 20]
    assert not v[ch, 0, 0] and not v[ch, 60, 70]   # far away: still unknown
    assert not v[1].any() and not v[2].any()       # other channels unaffected (only black spot verified)


def test_negative_roi_scope_ignore_unknown_rules(project, frame):
    store = AnnotationStore(project)
    shape = (frame["height"], frame["width"])
    store.add(frame["id"], "Normal Embossing", "negative", "bbox", {"x0": 0, "y0": 0, "x1": 10, "y1": 10})
    store.add(frame["id"], "Unknown", "ignore", "bbox", {"x0": 5, "y0": 5, "x1": 15, "y1": 15})
    store.add(frame["id"], "Dust", "positive", "bbox", {"x0": 40, "y0": 40, "x1": 50, "y1": 50})  # verdict undecided
    t, v, _ = build_targets(store, frame["id"], shape)
    assert v[:, 2, 2].all() and (t[:, 2, 2] == 0).all()     # verified negative for all defect channels
    assert not v[:, 7, 7].any()                              # ignore overrides negative
    store.set_review_scope(frame["id"], "Black Spot")
    t, v, _ = build_targets(store, frame["id"], shape)
    assert v[0, 30, 70] and not v[1, 30, 70]                 # exhaustive review only for black spot channel
    assert not v[0, 45, 45]                                  # undecided dust hides defect channels
    assert not v[0, 7, 7]


def test_masked_loss_ignores_unknown_region():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.zeros(2, 3, 16, 16); target[:, 0, 4, 4] = 1
    valid = torch.zeros(2, 3, 16, 16, dtype=torch.bool); valid[:, :, :8, :8] = True
    for fn in (masked_bce, masked_focal, masked_dice, masked_heatmap_focal, masked_mse):
        l1 = fn(logits, target, valid)
        logits2 = logits.detach().clone(); logits2[:, :, 8:, 8:] += 100.0   # change only unknown pixels
        l2 = fn(logits2, target, valid)
        assert torch.isclose(l1, l2), fn.__name__
        logits.grad = None
        l1.backward()
        g = logits.grad
        assert torch.all(g[:, :, 8:, :] == 0) and torch.all(g[:, :, :, 8:] == 0), fn.__name__


def test_batch_without_valid_pixels_is_safe():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.rand(2, 3, 8, 8)
    valid = torch.zeros_like(target, dtype=torch.bool)
    loss, parts = combined_loss(logits, target, valid, {"black_spot": "heatmap_focal", "bubble": "bce+dice",
                                                        "intensity": "focal"}, ["black_spot", "bubble", "intensity"])
    assert float(loss) == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.all(logits.grad == 0)
    assert all(v == 0 for k, v in parts.items() if k.endswith("valid_px"))


def test_labels_persist_across_reopen_with_history_and_undo(project, frame, tmp_path):
    store = AnnotationStore(project, author="tester")
    a1 = store.add(frame["id"], "Black Spot", "positive", "point", {"x": 3.5, "y": 4.5, "r": 1.5}, visibility="weak")
    a2 = store.add(frame["id"], "Verified Normal", "negative", "polygon", {"points": [[1, 1], [10, 1], [10, 9]]})
    mask = np.zeros((5, 7), np.uint8); mask[1:4, 2:6] = 1
    a3 = store.add(frame["id"], "Bubble", "positive", "mask", {"x0": 30, "y0": 20}, mask=mask)
    store.update(a1, visibility="visible")
    store.delete(a2)
    assert store.undo()                               # restores a2
    did = store.create_defect(frame["bottle_id"], "Black Spot")
    store.update(a1, defect_id=did, is_keyframe=1)
    root = project.root
    project.close()
    p2 = Project(root)
    s2 = AnnotationStore(p2)
    anns = {a["id"]: a for a in s2.for_frame(frame["id"])}
    assert set(anns) == {a1, a2, a3}
    assert anns[a1]["visibility"] == "visible" and anns[a1]["defect_id"] == did and anns[a1]["is_keyframe"] == 1
    assert anns[a1]["geometry"] == {"x": 3.5, "y": 4.5, "r": 1.5}
    assert np.array_equal(s2.load_mask(anns[a3]), mask)
    hist = [h["action"] for h in s2.history(a2)]
    assert hist == ["create", "delete", "undo-delete"]
    p2.close()


def test_ai_candidates_are_not_ground_truth(project, frame):
    store = AnnotationStore(project)
    run = project.create_run("classical", "test.alg")
    cid = save_candidates(project, run["id"], frame["id"], [
        Candidate(frame["id"], "m", "test.alg", "Black Spot", 9.0, "point", {"x": 10.5, "y": 10.5}),
        Candidate(frame["id"], "m", "test.alg", "Black Spot", 8.0, "point", {"x": 50.5, "y": 30.5})], 1.0)
    assert store.for_frame(frame["id"]) == []         # candidates do not appear as labels
    t, v, _ = build_targets(store, frame["id"], (frame["height"], frame["width"]))
    assert not v.any()                                 # nor in training targets
    a = store.accept_candidate(cid[0])
    n = store.reject_candidate(cid[1])
    acc, rej = store.get(a), store.get(n)
    assert acc["label_state"] == "positive" and acc["source"] == "ai_accepted"
    assert rej["label_state"] == "negative" and rej["attributes"]["hard_negative"]
    st = {r["id"]: r["review_status"] for r in project.db.query("SELECT id, review_status FROM candidates")}
    assert st == {cid[0]: "accepted", cid[1]: "rejected"}


def test_propagated_labels_unconfirmed_until_reviewed(project, frame):
    store = AnnotationStore(project)
    frames = project.all_frames()
    a = store.add(frames[0]["id"], "Black Spot", "positive", "point", {"x": 20.5, "y": 20.5})
    (b,) = store.propagate(a, [frames[1]["id"]])
    assert store.get(b)["confirmed"] == 0 and store.get(b)["visibility"] == "uncertain"
    t, v, _ = build_targets(store, frames[1]["id"], (frames[1]["height"], frames[1]["width"]))
    assert not v.any()                                 # unconfirmed copy is not GT
    store.confirm(b)
    t, v, _ = build_targets(store, frames[1]["id"], (frames[1]["height"], frames[1]["width"]))
    assert v[0, 20, 20] and t[0, 20, 20] == 1.0


def test_heatmap_area_is_not_physical_size():
    from gistudio.core.geometry import gaussian_heatmap
    hm1 = gaussian_heatmap((40, 40), [(20.3, 20.7)], sigma=1.0)
    hm3 = gaussian_heatmap((40, 40), [(20.3, 20.7)], sigma=3.0)
    assert hm1[20, 20] == 1.0 and hm3[20, 20] == 1.0  # peak pixel always 1
    assert (hm3 > 0.5).sum() > (hm1 > 0.5).sum()      # footprint depends only on sigma (a training choice)
