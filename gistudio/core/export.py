"""Export of the internal annotation format (keeps partial-label states, visibility, defect IDs)."""
from __future__ import annotations

import json

from .annotations import AnnotationStore
from .project import Project


def export_project_annotations(project: Project, path: str) -> int:
    store = AnnotationStore(project)
    out = {"format": "gistudio-annotations/1",
           "classes": project.classes(False),
           "defects": [dict(r) for r in project.db.query("SELECT * FROM defects")],
           "review_scopes": [dict(r) for r in project.db.query("SELECT * FROM review_scopes")],
           "frames": []}
    n = 0
    for f in project.all_frames():
        anns = store.for_frame(f["id"])
        n += len(anns)
        out["frames"].append({"frame_id": f["id"], "path": f["path"], "sha256": f["sha256"], "bottle": f["bottle_code"],
                              "session": f["session_code"], "angle_deg": f["angle_deg"],
                              "review_status": f["review_status"], "annotations": anns})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    return n
