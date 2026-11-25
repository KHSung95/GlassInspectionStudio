"""Project: a directory with studio.db + caches + experiment outputs.

Original image files are referenced by absolute path and are never written to.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .db import Database

DEFAULT_CLASSES = [
    # name, kind, color, shortcut, train_channel
    ("Black Spot", "defect", "#ff3030", "1", "black_spot"),
    ("Bubble", "defect", "#ff9900", "2", "bubble"),
    ("Open Bubble", "defect", "#ffbb33", "3", "bubble"),
    ("Closed Bubble", "defect", "#ffd966", "4", "bubble"),
    ("Intensity / Transparency Defect", "defect", "#cc33ff", "5", "intensity"),
    ("Other Defect", "defect", "#ff66aa", "6", ""),
    ("Normal Embossing", "normal", "#33cc66", "7", ""),
    ("Normal Engraving", "normal", "#33aa99", "8", ""),
    ("Normal Mold Pattern", "normal", "#2299dd", "9", ""),
    ("Dust", "appearance", "#bbbbbb", "D", ""),
    ("Scratch", "appearance", "#88aacc", "S", ""),
    ("Verified Normal", "normal", "#00ff88", "N", ""),
    ("Unknown", "meta", "#777777", "U", ""),
]

# Supervised output channels (U-Net). Order matters (checkpoint metadata stores it).
TRAIN_CHANNELS = ["black_spot", "bubble", "intensity"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_uid(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class Project:
    DB_NAME = "studio.db"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        if not (self.root / self.DB_NAME).exists():
            raise FileNotFoundError(f"Not a Glass Inspection Studio project: {self.root}")
        self.db = Database(self.root / self.DB_NAME)

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: str | os.PathLike, name: str | None = None) -> "Project":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / cls.DB_NAME).exists():
            raise FileExistsError(f"Project already exists at {root} (refusing to overwrite)")
        for d in ("cache/thumbs", "masks", "runs", "exports", "presets", "prompts"):
            (root / d).mkdir(parents=True, exist_ok=True)
        (root / "project.json").write_text(
            json.dumps({"name": name or root.name, "created_at": now_iso(), "format": "gistudio-project/1"}, indent=2),
            encoding="utf-8",
        )
        Database(root / cls.DB_NAME).close()
        p = cls(root)
        p._seed_classes()
        return p

    @classmethod
    def open_or_create(cls, root: str | os.PathLike) -> "Project":
        root = Path(root)
        return cls(root) if (root / cls.DB_NAME).exists() else cls.create(root)

    def close(self) -> None:
        self.db.close()

    def _seed_classes(self) -> None:
        for i, (name, kind, color, sc, ch) in enumerate(DEFAULT_CLASSES):
            self.db.execute(
                "INSERT OR IGNORE INTO classes(name, kind, color, shortcut, train_channel, sort_order) VALUES(?,?,?,?,?,?)",
                (name, kind, color, sc, ch, i),
            )
        self.db.commit()

    # ------------------------------------------------------------------ paths
    @property
    def name(self) -> str:
        try:
            return json.loads((self.root / "project.json").read_text(encoding="utf-8"))["name"]
        except Exception:
            return self.root.name

    def runs_dir(self) -> Path:
        return self.root / "runs"

    def masks_dir(self) -> Path:
        return self.root / "masks"

    def thumbs_dir(self) -> Path:
        return self.root / "cache" / "thumbs"

    # ------------------------------------------------------------------ classes
    def classes(self, active_only: bool = True) -> list[dict]:
        q = "SELECT * FROM classes" + (" WHERE active=1" if active_only else "") + " ORDER BY sort_order, id"
        return [dict(r) for r in self.db.query(q)]

    def class_map(self) -> dict[str, dict]:
        return {c["name"]: c for c in self.classes(active_only=False)}

    def upsert_class(self, name: str, kind: str, color: str = "#ffffff", shortcut: str = "",
                     train_channel: str = "", active: bool = True) -> None:
        if kind not in ("defect", "normal", "appearance", "meta"):
            raise ValueError(kind)
        self.db.execute(
            """INSERT INTO classes(name, kind, color, shortcut, train_channel, active, sort_order)
               VALUES(?,?,?,?,?,?, (SELECT COALESCE(MAX(sort_order),0)+1 FROM classes))
               ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, color=excluded.color, shortcut=excluded.shortcut,
               train_channel=excluded.train_channel, active=excluded.active""",
            (name, kind, color, shortcut, train_channel, int(active)),
        )
        self.db.commit()

    # ------------------------------------------------------------------ hierarchy
    def ensure_dataset(self, name: str, description: str = "") -> int:
        r = self.db.one("SELECT id FROM datasets WHERE name=?", (name,))
        if r:
            return r["id"]
        cur = self.db.execute("INSERT INTO datasets(name, description, created_at) VALUES(?,?,?)",
                              (name, description, now_iso()))
        self.db.commit()
        return cur.lastrowid

    def ensure_bottle(self, dataset_id: int, bottle_code: str, **fields) -> int:
        r = self.db.one("SELECT id FROM bottles WHERE dataset_id=? AND bottle_code=?", (dataset_id, bottle_code))
        if r:
            if fields:
                self.update_bottle(r["id"], **fields)
            return r["id"]
        fields.setdefault("physical_sample_id", bottle_code)
        cols = ["dataset_id", "bottle_code"] + list(fields)
        cur = self.db.execute(
            f"INSERT INTO bottles({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            [dataset_id, bottle_code] + list(fields.values()),
        )
        self.db.commit()
        return cur.lastrowid

    _BOTTLE_FIELDS = {"product_type", "variety", "lot_no", "physical_sample_id", "verdict", "memo"}
    _SESSION_FIELDS = {"captured_at", "camera_model", "bit_depth", "exposure_us", "gain_db", "lighting", "lens",
                       "magnification", "mm_per_px", "source_dir", "memo"}

    def update_bottle(self, bottle_id: int, **fields) -> None:
        bad = set(fields) - self._BOTTLE_FIELDS
        if bad:
            raise KeyError(bad)
        if fields:
            self.db.execute(f"UPDATE bottles SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                            list(fields.values()) + [bottle_id])
            self.db.commit()

    def ensure_session(self, bottle_id: int, session_code: str, **fields) -> int:
        r = self.db.one("SELECT id FROM sessions WHERE bottle_id=? AND session_code=?", (bottle_id, session_code))
        if r:
            if fields:
                self.update_session(r["id"], **fields)
            return r["id"]
        cols = ["bottle_id", "session_code"] + list(fields)
        cur = self.db.execute(
            f"INSERT INTO sessions({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            [bottle_id, session_code] + list(fields.values()),
        )
        self.db.commit()
        return cur.lastrowid

    def update_session(self, session_id: int, **fields) -> None:
        bad = set(fields) - self._SESSION_FIELDS
        if bad:
            raise KeyError(bad)
        if fields:
            self.db.execute(f"UPDATE sessions SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                            list(fields.values()) + [session_id])
            self.db.commit()

    def datasets(self) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM datasets ORDER BY id")]

    def bottles(self, dataset_id: int | None = None, search: str = "", product_type: str = "",
                verdict: str = "") -> list[dict]:
        q, p = "SELECT * FROM bottles WHERE 1=1", []
        if dataset_id is not None:
            q += " AND dataset_id=?"; p.append(dataset_id)
        if search:
            q += " AND (bottle_code LIKE ? OR lot_no LIKE ? OR memo LIKE ? OR physical_sample_id LIKE ?)"
            p += [f"%{search}%"] * 4
        if product_type:
            q += " AND product_type=?"; p.append(product_type)
        if verdict:
            q += " AND verdict=?"; p.append(verdict)
        return [dict(r) for r in self.db.query(q + " ORDER BY bottle_code", p)]

    def bottle(self, bottle_id: int) -> dict:
        return dict(self.db.one("SELECT * FROM bottles WHERE id=?", (bottle_id,)))

    def sessions(self, bottle_id: int) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM sessions WHERE bottle_id=? ORDER BY session_code",
                                               (bottle_id,))]

    def session(self, session_id: int) -> dict:
        return dict(self.db.one("SELECT * FROM sessions WHERE id=?", (session_id,)))

    def frames(self, session_id: int) -> list[dict]:
        return [dict(r) for r in self.db.query(
            "SELECT * FROM frames WHERE session_id=? ORDER BY COALESCE(angle_deg, 1e9), frame_index", (session_id,))]

    def frame(self, frame_id: int) -> dict:
        r = self.db.one(
            """SELECT f.*, s.bottle_id, s.session_code, s.bit_depth, s.mm_per_px, b.bottle_code, b.product_type,
                      b.physical_sample_id, b.dataset_id
               FROM frames f JOIN sessions s ON f.session_id=s.id JOIN bottles b ON s.bottle_id=b.id
               WHERE f.id=?""", (frame_id,))
        if r is None:
            raise KeyError(frame_id)
        return dict(r)

    def all_frames(self, bottle_ids: list[int] | None = None) -> list[dict]:
        q = """SELECT f.*, s.bottle_id, s.session_code, s.bit_depth, b.bottle_code, b.product_type, b.physical_sample_id
               FROM frames f JOIN sessions s ON f.session_id=s.id JOIN bottles b ON s.bottle_id=b.id"""
        p: list = []
        if bottle_ids is not None:
            if not bottle_ids:
                return []
            q += f" WHERE s.bottle_id IN ({','.join('?' * len(bottle_ids))})"
            p = list(bottle_ids)
        return [dict(r) for r in self.db.query(q + " ORDER BY b.bottle_code, s.session_code, f.angle_deg, f.frame_index", p)]

    def set_frame_review_status(self, frame_id: int, status: str) -> None:
        if status not in ("unreviewed", "in_progress", "reviewed"):
            raise ValueError(status)
        self.db.execute("UPDATE frames SET review_status=? WHERE id=?", (status, frame_id))
        self.db.commit()

    def product_types(self) -> list[str]:
        return [r[0] for r in self.db.query("SELECT DISTINCT product_type FROM bottles ORDER BY 1")]

    def stats(self) -> dict:
        one = lambda q: self.db.one(q)[0]  # noqa: E731
        return {
            "datasets": one("SELECT COUNT(*) FROM datasets"),
            "bottles": one("SELECT COUNT(*) FROM bottles"),
            "sessions": one("SELECT COUNT(*) FROM sessions"),
            "frames": one("SELECT COUNT(*) FROM frames"),
            "annotations": one("SELECT COUNT(*) FROM annotations WHERE deleted=0"),
            "confirmed_positive": one(
                "SELECT COUNT(*) FROM annotations WHERE deleted=0 AND confirmed=1 AND label_state='positive'"),
            "candidates": one("SELECT COUNT(*) FROM candidates"),
            "runs": one("SELECT COUNT(*) FROM runs"),
            "ok_bottles": one("SELECT COUNT(*) FROM bottles WHERE verdict='ok'"),
        }

    # ------------------------------------------------------------------ runs
    def create_run(self, kind: str, algorithm: str, params: dict | None = None, preprocess: dict | None = None,
                   model_info: dict | None = None, score_semantics: str = "", notes: str = "") -> dict:
        uid = new_uid(kind)
        out = self.runs_dir() / uid
        out.mkdir(parents=True, exist_ok=True)
        cur = self.db.execute(
            """INSERT INTO runs(run_uid, kind, algorithm, params, preprocess, model_info, score_semantics, status,
               output_dir, created_at, notes) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (uid, kind, algorithm, json.dumps(params or {}), json.dumps(preprocess or {}),
             json.dumps(model_info or {}), score_semantics, "running", str(out.relative_to(self.root)), now_iso(), notes),
        )
        self.db.commit()
        rec = {"id": cur.lastrowid, "run_uid": uid, "output_dir": out, "kind": kind, "algorithm": algorithm,
               "params": params or {}, "preprocess": preprocess or {}, "model_info": model_info or {}}
        (out / "run.json").write_text(json.dumps({**rec, "output_dir": str(out)}, indent=2, default=str),
                                      encoding="utf-8")
        return rec

    def finish_run(self, run_id: int, status: str = "done", notes: str | None = None) -> None:
        if notes is None:
            self.db.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?", (status, now_iso(), run_id))
        else:
            self.db.execute("UPDATE runs SET status=?, finished_at=?, notes=? WHERE id=?",
                            (status, now_iso(), notes, run_id))
        self.db.commit()

    def runs(self, kind: str | None = None) -> list[dict]:
        q, p = "SELECT * FROM runs", []
        if kind:
            q += " WHERE kind=?"; p.append(kind)
        return [dict(r) for r in self.db.query(q + " ORDER BY id DESC", p)]

    def run(self, run_id: int) -> dict:
        return dict(self.db.one("SELECT * FROM runs WHERE id=?", (run_id,)))
