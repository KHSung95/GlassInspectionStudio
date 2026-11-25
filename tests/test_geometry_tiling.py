"""Coordinate transforms, tiling/stitching restoration, circular angle handling."""
import math

import numpy as np
import pytest

from gistudio.core import geometry as geo
from gistudio.core.tiling import Stitcher, make_tiles, pad_to


# ------------------------------------------------------------------ image <-> view coordinates
def test_view_transform_roundtrip_and_zoom_anchor():
    t = geo.ViewTransform(scale=2.5, offset_x=-13.0, offset_y=7.0)
    for x, y in [(0, 0), (10.25, 3.75), (1023.5, 767.5)]:
        vx, vy = t.image_to_view(x, y)
        assert t.view_to_image(vx, vy) == pytest.approx((x, y))
    # zooming about a view point keeps the image point under the cursor fixed
    ix, iy = t.view_to_image(400, 300)
    t.zoom_about(4.0, 400, 300)
    assert t.view_to_image(400, 300) == pytest.approx((ix, iy))
    assert t.scale == pytest.approx(10.0)


def test_pixel_index_convention():
    assert geo.pixel_index(0.0, 0.0) == (0, 0)
    assert geo.pixel_index(0.999, 5.5) == (0, 5)
    assert geo.pixel_index(1.0, 5.0) == (1, 5)


def test_qt_viewer_mapping_matches_image_pixels():
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication
    from gistudio.app.viewer import ImageViewer
    app = QApplication.instance() or QApplication([])
    v = ImageViewer(); v.resize(400, 300); v.show()
    raw = np.arange(200 * 100, dtype=np.uint16).reshape(100, 200)
    v.set_image(raw, (raw % 256).astype(np.uint8))
    for z in (0.5, 1.0, 7.0, 33.0):
        v.center_on_image(57.3, 41.8, z); app.processEvents()
        p = v.mapFromScene(QPointF(57.3, 41.8))
        s = v.mapToScene(p)
        assert abs(s.x() - 57.3) <= 1.0 / z + 1e-6 and abs(s.y() - 41.8) <= 1.0 / z + 1e-6
        assert v.zoom() == pytest.approx(z)
    v.close()


# ------------------------------------------------------------------ tiling
@pytest.mark.parametrize("h,w,tile,ov", [(100, 130, 32, 8), (64, 64, 64, 0), (17, 300, 64, 16), (10, 10, 32, 4),
                                         (513, 1025, 256, 64)])
def test_tiles_cover_image_and_restore_exactly(h, w, tile, ov):
    img = np.random.default_rng(0).random((h, w)).astype(np.float32)
    tiles = make_tiles(h, w, tile, ov)
    cover = np.zeros((h, w), int)
    for t in tiles:
        assert 0 <= t.x0 and t.x1 <= w and 0 <= t.y0 and t.y1 <= h
        cover[t.y0:t.y1, t.x0:t.x1] += 1
    assert cover.min() >= 1
    for mode, win in (("mean", False), ("mean", True), ("max", False)):
        st = Stitcher(h, w, 1, mode=mode, window=win)
        for t in tiles:
            st.add(t, img[t.y0:t.y1, t.x0:t.x1])
        assert np.allclose(st.result()[0], img, atol=1e-6)


def test_tile_local_coordinates_map_to_original():
    h, w = 300, 500
    tiles = make_tiles(h, w, 128, 32)
    gx, gy = 431.5, 211.5                              # an original point
    hits = [t for t in tiles if t.x0 <= gx < t.x1 and t.y0 <= gy < t.y1]
    assert hits
    for t in hits:
        lx, ly = gx - t.x0, gy - t.y0                  # tile-local
        assert (lx + t.x0, ly + t.y0) == (gx, gy)


def test_pad_to_keeps_original_region():
    a = np.arange(30, dtype=np.float32).reshape(5, 6)
    p, (h, w) = pad_to(a, 8)
    assert p.shape == (8, 8) and (h, w) == (5, 6) and np.array_equal(p[:5, :6], a)


# ------------------------------------------------------------------ 0/360 deg boundary
def test_circular_angles():
    assert geo.wrap_deg(360.0) == 0.0 and geo.wrap_deg(-1.0) == 359.0 and geo.wrap_deg(725) == 5.0
    assert geo.circ_dist(359.0, 1.0) == pytest.approx(2.0)
    assert geo.circ_diff(1.0, 359.0) == pytest.approx(2.0)
    assert geo.nearest_angle_index([0, 90, 180, 270], 350) == 0
    assert geo.nearest_angle_index([0, 90, 180, 270], -100) == 3


def test_visible_interval_crossing_zero_is_merged():
    angles = [i * 10.0 for i in range(36)]
    vis = [a >= 340 or a <= 20 for a in angles]
    iv = geo.circular_intervals(angles, vis, 10.0)
    assert len(iv) == 1
    s, e = iv[0]
    assert geo.wrap_deg(s) == pytest.approx(335.0) and e - s == pytest.approx(50.0)


def test_appstate_navigation_wraps(project, tmp_path):
    from conftest import make_frames
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from gistudio.app.state import AppState
    make_frames(project, tmp_path, n_bottles=1, n_frames=4)
    st = AppState(); st.project = project
    from gistudio.core.annotations import AnnotationStore
    st.store = AnnotationStore(project)
    st.set_session(project.sessions(project.bottles()[0]["id"])[0]["id"])
    angles = [f["angle_deg"] for f in st.frames]
    assert angles == [0.0, 90.0, 180.0, 270.0]
    st.goto_index(0); st.step(-1)
    assert st.frame["angle_deg"] == 270.0          # 0 -> previous wraps to 270
    st.step(1)
    assert st.frame["angle_deg"] == 0.0
    st.goto_angle(359.0)
    assert st.frame["angle_deg"] == 0.0            # nearest on the circle, not 270
    st.goto_angle(-85.0)
    assert st.frame["angle_deg"] == 270.0


def test_sampling_experiment_start_offsets():
    from gistudio.evaluation.angular import sampling_experiment
    rows = [{"angle": a * 1.0, "annotated": True, "score": 1.0 if 100 <= a < 110 else 0.0} for a in range(360)]
    prof = {"rows": rows}
    r = sampling_experiment([prof], steps=(1, 5, 10, 30), n_starts=10, key="score", thr=0.5)
    assert r["1"]["worst_miss_rate"] == 0.0
    assert r["5"]["worst_miss_rate"] == 0.0       # 10 deg window always hit with 5 deg step
    assert r["30"]["mean_miss_rate"] > 0.0        # coarse step misses for some start angles
    assert 0.0 <= r["30"]["mean_miss_rate"] <= r["30"]["worst_miss_rate"] <= 1.0
