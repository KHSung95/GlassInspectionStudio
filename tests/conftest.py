import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistudio.core.project import Project  # noqa: E402


@pytest.fixture
def project(tmp_path):
    p = Project.create(tmp_path / "proj")
    yield p
    p.close()


def make_frames(project, tmp_path, n_bottles=2, n_frames=4, shape=(64, 80), physical=None, ptype="A"):
    """Create tiny uint16 TIFF frames and register them via the importer."""
    import tifffile
    from gistudio.core.dataset_import import ImportOptions, import_folder
    root = tmp_path / "imgs"
    rng = np.random.default_rng(0)
    for b in range(n_bottles):
        d = root / f"B{b}" / "S001"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            a = rng.integers(1000, 3000, size=shape, dtype=np.uint16)
            tifffile.imwrite(d / f"img_a{i * 90 / max(1, n_frames // 4)}_{i:03d}.tif", a)
    rep = import_folder(project, root, ImportOptions(layout="bottle/session", dataset_name="t"))
    for b in project.bottles():
        project.update_bottle(b["id"], product_type=ptype,
                              physical_sample_id=(physical or {}).get(b["bottle_code"], b["bottle_code"]))
    return rep
