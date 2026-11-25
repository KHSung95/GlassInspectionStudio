"""Parameter sweep recommends presets from labels; ROI scores match full-frame scores."""
import json

import numpy as np
import tifffile

from gistudio.algorithms.base import create
from gistudio.core.annotations import AnnotationStore
from gistudio.core.dataset_import import ImportOptions, import_folder
from gistudio.evaluation.param_sweep import SweepConfig, labelled_frames, run_sweep, save_presets


def _scene(tmp_path, project, n=3):
    rng = np.random.default_rng(0)
    d = tmp_path / "raw" / "B1" / "S1"; d.mkdir(parents=True)
    spots = []
    for i in range(n):
        img = (2000 + rng.normal(0, 20, (240, 320))).astype(np.uint16)
        img[100:102, 40:200] = 1200                           # normal 'graduation' line (hard negative)
        pts = [(60 + 70 * k + 5 * i, 50 + 40 * k) for k in range(3)]
        for x, y in pts:
            img[y - 1:y + 2, x - 1:x + 2] = 1300               # 3x3 dark spots
        tifffile.imwrite(d / f"f_a{i * 10}.tif", img)
        spots.append(pts)
    import_folder(project, tmp_path / "raw", ImportOptions())
    store = AnnotationStore(project)
    for f, pts in zip(project.all_frames(), spots):
        for x, y in pts:
            store.add(f["id"], "Black Spot", "positive", "point", {"x": x + 0.5, "y": y + 0.5})
        store.add(f["id"], "Normal Engraving", "negative", "bbox", {"x0": 30, "y0": 95, "x1": 210, "y1": 107})
    return spots


def test_roi_scores_equal_full_frame_with_frame_sigma(project, tmp_path):
    _scene(tmp_path, project, 1)
    from gistudio.algorithms.classical.ops import noise_sigma
    from gistudio.core.image_io import load_mono
    img = load_mono(project.all_frames()[0]["path"])
    alg = create("classical.blackhat", threshold=5.0)
    full = {(round(c.geometry["x"]), round(c.geometry["y"])): c.score for c in alg.run(img).candidates}
    roi = alg.run(img[20:200, 20:300], {"noise_sigma": noise_sigma(img)})
    compared = 0
    for c in roi.candidates:
        x, y = c.geometry["x"], c.geometry["y"]
        if not (15 <= x <= 280 - 15 and 15 <= y <= 180 - 15):   # morphology border effects near the ROI edge
            continue
        k = (round(x) + 20, round(y) + 20)
        if k in full:
            assert abs(full[k] - c.score) < 1e-3
            compared += 1
    assert compared >= 3


def test_sweep_recommends_presets(project, tmp_path):
    _scene(tmp_path, project, 3)
    fids = labelled_frames(project)
    assert len(fids) == 3
    # the graduation line is darker than the spots: only the area limit (max_area) can separate it
    r = run_sweep(project, SweepConfig("classical.blackhat", fids, {"kernel": [5, 9, 15], "max_area": [50, 400]},
                                       max_combos=10, threshold_grid=[6.0, 11.0, 22.0]))
    objs = {x["objective"]: x for x in r["recommendations"]}
    assert "balanced" in objs and "high_recall" in objs
    assert objs["balanced"]["tune"]["recall"] >= 0.9                 # clear 3x3 spots must be found
    assert objs["balanced"]["tune"]["fp_verified"] == 0             # line on verified-normal ROI rejected
    assert objs["balanced"]["params"]["max_area"] == 50
    assert r["n_gt"] == 9 and r["combos_evaluated"] == 6
    paths = save_presets(project, r)
    d = json.loads(open(paths[0], encoding="utf-8").read())
    assert d["algorithm"] == "classical.blackhat" and "threshold" in d["params"]
