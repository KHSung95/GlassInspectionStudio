"""Draft labelling server core: add (draft), confirm provisional, delete — all through AnnotationStore."""
import numpy as np
import tifffile

from gistudio.app.web_label import LabelServer
from gistudio.core.annotations import AnnotationStore
from gistudio.core.dataset_import import ImportOptions, import_folder


def test_label_server_roundtrip(project, tmp_path):
    d = tmp_path / "raw"; d.mkdir()
    tifffile.imwrite(d / "a.tif", np.full((40, 60), 100, np.uint8))
    import_folder(project, d, ImportOptions(layout="flat", bottle_code="B"))
    srv = LabelServer(project)
    assert "Marker Ink" in project.class_map()
    f = srv.frames()[0]
    assert srv.image(f["id"], "stretch")[:2] == b"\xff\xd8"
    a = srv.add(f["id"], {"class_name": "Black Spot", "geometry_type": "point", "geometry": {"x": 10.5, "y": 5.5}})
    assert a["attributes"]["draft"] and a["confirmed"] == 1 and a["label_state"] == "positive"
    prov = AnnotationStore(project, "claude-pilot").add(f["id"], "Bubble", "positive", "point", {"x": 30, "y": 20},
                                                        attributes={"provisional_by": "claude-pilot"})
    c = srv.update(prov, {"review": "confirm"})
    assert c["attributes"]["reviewed_by"] == "operator"
    srv.delete(a["id"])
    assert [x["id"] for x in srv.anns(f["id"])] == [prov]
    assert srv.frames()[0]["n_pos"] == 1
