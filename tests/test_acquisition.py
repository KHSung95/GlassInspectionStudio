"""Capture path with the mock camera (hardware-free): lossless write, immediate registration, no overwrite."""
import numpy as np
import pytest

from gistudio.acquisition.capture_session import CaptureSessionWriter, CaptureTarget
from gistudio.acquisition.mock import MockCamera, MockRotationStage
from gistudio.core.image_io import read_image


def test_capture_writer_registers_frames_losslessly(project):
    stage = MockRotationStage(); cam = MockCamera(stage, 160, 120); cam.open()
    w = CaptureSessionWriter(project, CaptureTarget("NG-001", "S1", "VIAL"), cam.info())
    ids, imgs = [], []
    for a in (0.0, 30.0, 350.0):
        stage.move_to(a)
        img = cam.grab(); imgs.append(img)
        ids.append(w.add(img, a))
    fr = project.frames(w.session_id)
    assert [f["id"] for f in fr] == ids and [f["angle_deg"] for f in fr] == [0.0, 30.0, 350.0]
    for f, img in zip(fr, imgs):
        assert np.array_equal(read_image(f["path"]), img) and f["dtype"] == "uint16"
    assert project.bottles()[0]["product_type"] == "VIAL"
    with pytest.raises(FileExistsError):
        CaptureSessionWriter(project, CaptureTarget("NG-001", "S1"), cam.info())   # never overwrite a session
    csv_lines = (w.dir / "angles.csv").read_text().splitlines()
    assert len(csv_lines) == 4
    cam.close()


def test_image_quality_metrics():
    from gistudio.app.panels_acquisition import image_quality
    sharp = np.zeros((200, 200), np.uint16); sharp[::2, ::2] = 4095
    blur = np.full((200, 200), 2000, np.uint16)
    qs, qb = image_quality(sharp, 12), image_quality(blur, 12)
    assert qs["focus_lapvar"] > qb["focus_lapvar"] and qs["saturated_pct"] > 0 and qb["saturated_pct"] == 0
