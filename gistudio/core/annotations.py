"""Annotation store with partial-label semantics, defect identity, history, undo/redo.

Label states
------------
positive : confirmed defect (of `class_name`)
negative : human-verified normal / acceptable region
unknown  : explicitly marked as undecidable (also the implicit state of every unlabelled pixel)
ignore   : region excluded from training AND evaluation (e.g. fixture, out-of-focus)

AI/algorithm outputs live in the `candidates` table and become annotations only through explicit review
(`accept_candidate` / `reject_candidate`). Propagated/tracked labels are stored with confirmed=0 and are
excluded from ground truth until a user confirms them.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .project import Project, now_iso

LABEL_STATES = ("positive", "negative", "unknown", "ignore")
GEOMETRY_TYPES = ("point", "bbox", "polygon", "mask")
VISIBILITY = ("visible", "weak", "not_visible", "uncertain")
SOURCES = ("manual", "copied", "interpolated", "tracked", "ai_accepted", "ai_rejected", "imported")


def _row(r) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    d["geometry"] = json.loads(d["geometry"])
    d["attributes"] = json.loads(d.get("attributes") or "{}")
    return d


@dataclass
class _Cmd:
    kind: str          # create | update | delete
    ann_id: int
    before: dict | None
    after: dict | None


class AnnotationStore:
    """All writes are committed immediately (auto-save). Undo/redo works on this store's command stack."""

    def __init__(self, project: Project, author: str = ""):
        self.p = project
        self.db = project.db
        self.author = author
        self._undo: list[_Cmd] = []
        self._redo: list[_Cmd] = []

    # ------------------------------------------------------------------ validation
    def _validate(self, class_name: str, label_state: str, gtype: str, geometry: dict, visibility: str) -> None:
        if label_state not in LABEL_STATES:
            raise ValueError(f"label_state must be one of {LABEL_STATES}")
        if gtype not in GEOMETRY_TYPES:
            raise ValueError(f"geometry_type must be one of {GEOMETRY_TYPES}")
        if visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}")
        if class_name not in self.p.class_map():
            raise KeyError(f"Unknown class '{class_name}'. Add it in the class editor first.")
        req = {"point": ("x", "y"), "bbox": ("x0", "y0", "x1", "y1"), "polygon": ("points",),
               "mask": ("x0", "y0", "w", "h")}[gtype]
        for k in req:
            if k not in geometry:
                raise ValueError(f"{gtype} geometry requires '{k}'")
        if gtype == "polygon" and len(geometry["points"]) < 3:
            raise ValueError("polygon needs >= 3 points")

    # ------------------------------------------------------------------ core CRUD (no undo recording)
    def _insert(self, d: dict) -> int:
        ts = now_iso()
        cur = self.db.execute(
            """INSERT INTO annotations(frame_id, defect_id, class_name, label_state, geometry_type, geometry, mask_path,
               visibility, is_keyframe, source, confirmed, verdict, attributes, author, created_at, updated_at, deleted)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (d["frame_id"], d.get("defect_id"), d["class_name"], d["label_state"], d["geometry_type"],
             json.dumps(d["geometry"]), d.get("mask_path"), d.get("visibility", "visible"),
             int(d.get("is_keyframe", 0)), d.get("source", "manual"), int(d.get("confirmed", 1)),
             d.get("verdict", "undecided"), json.dumps(d.get("attributes", {})), self.author, ts, ts),
        )
        aid = cur.lastrowid
        self._history(aid, "create", None, d)
        self.db.commit()
        return aid

    def _write(self, aid: int, d: dict, action: str = "update") -> None:
        before = self.get(aid, include_deleted=True)
        self.db.execute(
            """UPDATE annotations SET defect_id=?, class_name=?, label_state=?, geometry_type=?, geometry=?, mask_path=?,
               visibility=?, is_keyframe=?, source=?, confirmed=?, verdict=?, attributes=?, updated_at=?, deleted=?
               WHERE id=?""",
            (d.get("defect_id"), d["class_name"], d["label_state"], d["geometry_type"], json.dumps(d["geometry"]),
             d.get("mask_path"), d.get("visibility", "visible"), int(d.get("is_keyframe", 0)),
             d.get("source", "manual"), int(d.get("confirmed", 1)), d.get("verdict", "undecided"),
             json.dumps(d.get("attributes", {})), now_iso(), int(d.get("deleted", 0)), aid),
        )
        self._history(aid, action, before, d)
        self.db.commit()

    def _history(self, aid: int, action: str, before, after) -> None:
        self.db.execute(
            "INSERT INTO annotation_history(annotation_id, action, before, after, author, ts) VALUES(?,?,?,?,?,?)",
            (aid, action, json.dumps(before, default=str) if before else None,
             json.dumps(after, default=str) if after else None, self.author, now_iso()),
        )

    # ------------------------------------------------------------------ public API
    def add(self, frame_id: int, class_name: str, label_state: str, geometry_type: str, geometry: dict,
            defect_id: int | None = None, visibility: str = "visible", is_keyframe: bool = False,
            source: str = "manual", confirmed: bool = True, verdict: str = "undecided",
            attributes: dict | None = None, mask: np.ndarray | None = None) -> int:
        geometry = dict(geometry)
        mask_path = None
        if geometry_type == "mask":
            if mask is None:
                raise ValueError("mask geometry requires a mask array")
            geometry.setdefault("h", int(mask.shape[0]))
            geometry.setdefault("w", int(mask.shape[1]))
            mask_path = self._save_mask(mask)
        self._validate(class_name, label_state, geometry_type, geometry, visibility)
        d = dict(frame_id=frame_id, defect_id=defect_id, class_name=class_name, label_state=label_state,
                 geometry_type=geometry_type, geometry=geometry, mask_path=mask_path, visibility=visibility,
                 is_keyframe=int(is_keyframe), source=source, confirmed=int(confirmed), verdict=verdict,
                 attributes=attributes or {})
        aid = self._insert(d)
        self._undo.append(_Cmd("create", aid, None, self.get(aid)))
        self._redo.clear()
        st = self.db.one("SELECT review_status FROM frames WHERE id=?", (frame_id,))
        if st and st["review_status"] == "unreviewed":
            self.p.set_frame_review_status(frame_id, "in_progress")
        return aid

    def update(self, aid: int, **changes) -> None:
        before = self.get(aid)
        if before is None:
            raise KeyError(aid)
        after = {**before, **changes}
        self._validate(after["class_name"], after["label_state"], after["geometry_type"], after["geometry"],
                       after.get("visibility", "visible"))
        self._write(aid, after)
        self._undo.append(_Cmd("update", aid, before, self.get(aid)))
        self._redo.clear()

    def delete(self, aid: int) -> None:
        before = self.get(aid)
        if before is None:
            return
        self._write(aid, {**before, "deleted": 1}, action="delete")
        self._undo.append(_Cmd("delete", aid, before, None))
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        c = self._undo.pop()
        if c.kind == "create":
            self._write(c.ann_id, {**c.after, "deleted": 1}, action="undo-create")
        elif c.kind == "update":
            self._write(c.ann_id, c.before, action="undo-update")
        elif c.kind == "delete":
            self._write(c.ann_id, {**c.before, "deleted": 0}, action="undo-delete")
        self._redo.append(c)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        c = self._redo.pop()
        if c.kind == "create":
            self._write(c.ann_id, {**c.after, "deleted": 0}, action="redo-create")
        elif c.kind == "update":
            self._write(c.ann_id, c.after, action="redo-update")
        elif c.kind == "delete":
            self._write(c.ann_id, {**c.before, "deleted": 1}, action="redo-delete")
        self._undo.append(c)
        return True

    def get(self, aid: int, include_deleted: bool = False) -> dict | None:
        r = self.db.one("SELECT * FROM annotations WHERE id=?" + ("" if include_deleted else " AND deleted=0"), (aid,))
        return _row(r)

    def for_frame(self, frame_id: int, confirmed_only: bool = False) -> list[dict]:
        q = "SELECT * FROM annotations WHERE frame_id=? AND deleted=0"
        if confirmed_only:
            q += " AND confirmed=1"
        return [_row(r) for r in self.db.query(q + " ORDER BY id", (frame_id,))]

    def search(self, class_name: str = "", label_state: str = "", source: str = "", confirmed: int | None = None,
               bottle_id: int | None = None, limit: int = 5000) -> list[dict]:
        q = """SELECT a.*, f.angle_deg, f.session_id, s.bottle_id FROM annotations a
               JOIN frames f ON a.frame_id=f.id JOIN sessions s ON f.session_id=s.id WHERE a.deleted=0"""
        p: list = []
        for col, val in (("a.class_name", class_name), ("a.label_state", label_state), ("a.source", source)):
            if val:
                q += f" AND {col}=?"; p.append(val)
        if confirmed is not None:
            q += " AND a.confirmed=?"; p.append(int(confirmed))
        if bottle_id is not None:
            q += " AND s.bottle_id=?"; p.append(bottle_id)
        return [_row(r) for r in self.db.query(q + f" ORDER BY a.id LIMIT {int(limit)}", p)]

    def history(self, aid: int) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM annotation_history WHERE annotation_id=? ORDER BY id", (aid,))]

    def confirm(self, aid: int) -> None:
        self.update(aid, confirmed=1)

    # ------------------------------------------------------------------ masks
    def _save_mask(self, mask: np.ndarray) -> str:
        rel = f"masks/{uuid.uuid4().hex}.npz"
        np.savez_compressed(self.p.root / rel, mask=(mask > 0).astype(np.uint8))
        return rel

    def load_mask(self, ann: dict) -> np.ndarray | None:
        if not ann.get("mask_path"):
            return None
        return np.load(self.p.root / ann["mask_path"])["mask"]

    def rasterize(self, ann: dict, shape: tuple[int, int], point_radius: float | None = None) -> np.ndarray:
        return geo.rasterize(ann["geometry_type"], ann["geometry"], shape, self.load_mask(ann), point_radius)

    # ------------------------------------------------------------------ overlaps
    def overlaps(self, frame_id: int, shape: tuple[int, int]) -> list[tuple[int, int, str]]:
        """Pairs of annotations whose regions overlap with conflicting label_state (for QA display)."""
        anns = self.for_frame(frame_id)
        out = []
        boxes = [(a, geo.geometry_bbox(a["geometry_type"], a["geometry"])) for a in anns]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (a, ba), (b, bb) = boxes[i], boxes[j]
                if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
                    continue
                if (self.rasterize(a, shape) & self.rasterize(b, shape)).any():
                    kind = "conflict" if a["label_state"] != b["label_state"] else "overlap"
                    out.append((a["id"], b["id"], kind))
        return out

    # ------------------------------------------------------------------ review scopes
    def set_review_scope(self, frame_id: int, class_name: str, exhaustive: bool = True) -> None:
        """Mark that `class_name` was exhaustively reviewed on this frame: unlabelled pixels become negatives
        *for that class only*."""
        if exhaustive:
            self.db.execute("INSERT OR REPLACE INTO review_scopes(frame_id, class_name, author, ts) VALUES(?,?,?,?)",
                            (frame_id, class_name, self.author, now_iso()))
        else:
            self.db.execute("DELETE FROM review_scopes WHERE frame_id=? AND class_name=?", (frame_id, class_name))
        self.db.commit()

    def review_scopes(self, frame_id: int) -> list[str]:
        return [r[0] for r in self.db.query("SELECT class_name FROM review_scopes WHERE frame_id=?", (frame_id,))]

    # ------------------------------------------------------------------ defects / tracking
    def create_defect(self, bottle_id: int, class_name: str, keyframe_id: int | None = None, memo: str = "") -> int:
        uid = f"D-{uuid.uuid4().hex[:8]}"
        cur = self.db.execute(
            "INSERT INTO defects(bottle_id, defect_uid, class_name, keyframe_id, memo, created_at) VALUES(?,?,?,?,?,?)",
            (bottle_id, uid, class_name, keyframe_id, memo, now_iso()))
        self.db.commit()
        return cur.lastrowid

    def update_defect(self, defect_id: int, **fields) -> None:
        ok = {"class_name", "final_verdict", "open_state", "keyframe_id", "memo"}
        if set(fields) - ok:
            raise KeyError(set(fields) - ok)
        if "final_verdict" in fields and fields["final_verdict"] not in ("undecided", "ng", "ok"):
            raise ValueError("final_verdict")
        if "open_state" in fields and fields["open_state"] not in ("open", "closed", "undetermined"):
            raise ValueError("open_state")
        self.db.execute(f"UPDATE defects SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                        list(fields.values()) + [defect_id])
        self.db.commit()

    def defects(self, bottle_id: int) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM defects WHERE bottle_id=? ORDER BY id", (bottle_id,))]

    def defect_observations(self, defect_id: int) -> list[dict]:
        q = """SELECT a.*, f.angle_deg, f.frame_index FROM annotations a JOIN frames f ON a.frame_id=f.id
               WHERE a.defect_id=? AND a.deleted=0 ORDER BY f.angle_deg, f.frame_index"""
        return [_row(r) for r in self.db.query(q, (defect_id,))]

    def propagate(self, ann_id: int, target_frame_ids: list[int], mode: str = "copy",
                  end_ann_id: int | None = None) -> list[int]:
        """Copy (or linearly interpolate between two keyframes) a label to other frames.

        Results are saved as confirmed=0 / visibility='uncertain' — they are suggestions and are excluded from
        ground truth until the user confirms each one, because refraction makes the motion non-linear and the
        defect may be invisible at some angles.
        """
        src = self.get(ann_id)
        if src is None:
            raise KeyError(ann_id)
        end = self.get(end_ann_id) if end_ann_id else None
        out = []
        n = len(target_frame_ids)
        for k, fid in enumerate(target_frame_ids):
            g = dict(src["geometry"])
            if mode == "interpolate" and end is not None and src["geometry_type"] == end["geometry_type"] == "point":
                t = (k + 1) / (n + 1)
                g["x"] = src["geometry"]["x"] * (1 - t) + end["geometry"]["x"] * t
                g["y"] = src["geometry"]["y"] * (1 - t) + end["geometry"]["y"] * t
            elif mode == "interpolate" and end is not None and src["geometry_type"] == end["geometry_type"] == "bbox":
                t = (k + 1) / (n + 1)
                for key in ("x0", "y0", "x1", "y1"):
                    g[key] = src["geometry"][key] * (1 - t) + end["geometry"][key] * t
            mask = self.load_mask(src) if src["geometry_type"] == "mask" else None
            out.append(self.add(fid, src["class_name"], src["label_state"], src["geometry_type"], g,
                                defect_id=src["defect_id"], visibility="uncertain",
                                source="interpolated" if mode == "interpolate" else "copied",
                                confirmed=False, verdict=src["verdict"], mask=mask))
        return out

    # ------------------------------------------------------------------ candidates <-> GT
    def accept_candidate(self, cand_id: int, class_name: str | None = None, defect_id: int | None = None,
                         as_point: bool | None = None) -> int:
        c = self.db.one("SELECT * FROM candidates WHERE id=?", (cand_id,))
        if c is None:
            raise KeyError(cand_id)
        g = json.loads(c["geometry"])
        gtype = c["geometry_type"]
        if gtype == "mask" or as_point:
            cx, cy = geo.geometry_center(gtype, g)
            gtype, g = "point", {"x": cx, "y": cy, "r": 2.0}
        aid = self.add(c["frame_id"], class_name or c["class_name"], "positive", gtype, g, defect_id=defect_id,
                       source="ai_accepted", attributes={"candidate_id": cand_id, "model_id": c["model_id"],
                                                         "score": c["score"]})
        self.db.execute("UPDATE candidates SET review_status='accepted', linked_annotation_id=? WHERE id=?",
                        (aid, cand_id))
        self.db.commit()
        return aid

    def reject_candidate(self, cand_id: int, normal_class: str = "Verified Normal", hard_negative: bool = True,
                         radius: float = 4.0) -> int | None:
        """Reject a candidate. If hard_negative, the region is stored as a verified negative (Hard Negative Mining)."""
        c = self.db.one("SELECT * FROM candidates WHERE id=?", (cand_id,))
        if c is None:
            raise KeyError(cand_id)
        aid = None
        if hard_negative:
            g = json.loads(c["geometry"])
            gtype = c["geometry_type"]
            if gtype in ("point", "mask"):
                cx, cy = geo.geometry_center(gtype, g)
                gtype, g = "bbox", {"x0": cx - radius, "y0": cy - radius, "x1": cx + radius, "y1": cy + radius}
            aid = self.add(c["frame_id"], normal_class, "negative", gtype, g, source="ai_rejected",
                           attributes={"candidate_id": cand_id, "model_id": c["model_id"], "score": c["score"],
                                       "hard_negative": True})
        self.db.execute("UPDATE candidates SET review_status='rejected', linked_annotation_id=? WHERE id=?",
                        (aid, cand_id))
        self.db.commit()
        return aid

    def mark_candidate_unknown(self, cand_id: int) -> None:
        self.db.execute("UPDATE candidates SET review_status='unknown' WHERE id=?", (cand_id,))
        self.db.commit()
