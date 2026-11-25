"""SQLite schema and connection management.

All project metadata, annotations, candidates and experiment records live in one
SQLite file (``studio.db``) inside the project directory. Original images are
referenced by path and are never modified.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = r"""
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bottles (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    bottle_code TEXT NOT NULL,
    product_type TEXT DEFAULT '',
    variety TEXT DEFAULT '',
    lot_no TEXT DEFAULT '',
    physical_sample_id TEXT DEFAULT '',
    verdict TEXT DEFAULT 'unknown',          -- customer/bottle-level status: unknown | ng | ok | under_review
    memo TEXT DEFAULT '',
    UNIQUE(dataset_id, bottle_code)
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    bottle_id INTEGER NOT NULL REFERENCES bottles(id),
    session_code TEXT NOT NULL,
    captured_at TEXT DEFAULT '',
    camera_model TEXT DEFAULT '',
    bit_depth INTEGER,                        -- effective sensor bit depth (10/12/16...), may differ from container
    exposure_us REAL,
    gain_db REAL,
    lighting TEXT DEFAULT '',
    lens TEXT DEFAULT '',
    magnification REAL,
    mm_per_px REAL,
    source_dir TEXT DEFAULT '',
    memo TEXT DEFAULT '',
    UNIQUE(bottle_id, session_code)
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    frame_index INTEGER NOT NULL,
    angle_deg REAL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    dtype TEXT,
    container_bits INTEGER,
    captured_at TEXT DEFAULT '',
    review_status TEXT DEFAULT 'unreviewed',  -- unreviewed | in_progress | reviewed
    memo TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_frames_session ON frames(session_id, frame_index);
CREATE INDEX IF NOT EXISTS ix_frames_sha ON frames(sha256);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,                       -- defect | normal | appearance | meta
    color TEXT NOT NULL,
    shortcut TEXT DEFAULT '',
    train_channel TEXT DEFAULT '',            -- supervised output channel this class maps to ('' = none)
    active INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0
);

-- Physical defect identity shared across frames/angles.
CREATE TABLE IF NOT EXISTS defects (
    id INTEGER PRIMARY KEY,
    bottle_id INTEGER NOT NULL REFERENCES bottles(id),
    defect_uid TEXT NOT NULL UNIQUE,
    class_name TEXT NOT NULL,
    final_verdict TEXT DEFAULT 'undecided',   -- undecided | ng | ok  (customer-spec decision, separate from appearance)
    open_state TEXT DEFAULT 'undetermined',   -- open | closed | undetermined (bubbles)
    keyframe_id INTEGER,
    memo TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id),
    defect_id INTEGER REFERENCES defects(id),
    class_name TEXT NOT NULL,
    label_state TEXT NOT NULL,                -- positive | negative | unknown | ignore
    geometry_type TEXT NOT NULL,              -- point | bbox | polygon | mask
    geometry TEXT NOT NULL,                   -- JSON in original image pixel coordinates
    mask_path TEXT,                           -- relative path (project dir) for brush masks
    visibility TEXT DEFAULT 'visible',        -- visible | weak | not_visible | uncertain
    is_keyframe INTEGER DEFAULT 0,
    source TEXT DEFAULT 'manual',             -- manual | copied | interpolated | tracked | ai_accepted | imported
    confirmed INTEGER DEFAULT 1,              -- 0 for propagated/tracked labels awaiting review (excluded from GT)
    verdict TEXT DEFAULT 'undecided',         -- per-instance spec decision (dust/scratch etc.)
    attributes TEXT DEFAULT '{}',
    author TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ann_frame ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS ix_ann_defect ON annotations(defect_id);

CREATE TABLE IF NOT EXISTS annotation_history (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL,
    action TEXT NOT NULL,                     -- create | update | delete | restore
    before TEXT,
    after TEXT,
    author TEXT DEFAULT '',
    ts TEXT NOT NULL
);

-- Class-scoped exhaustive review: when present, every pixel of the frame that is not
-- marked positive for `class_name` is a verified negative for that class.
CREATE TABLE IF NOT EXISTS review_scopes (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id),
    class_name TEXT NOT NULL,
    author TEXT DEFAULT '',
    ts TEXT NOT NULL,
    UNIQUE(frame_id, class_name)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_uid TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,                       -- classical | zeroshot | train | infer | eval | angular
    algorithm TEXT NOT NULL,
    params TEXT DEFAULT '{}',
    preprocess TEXT DEFAULT '{}',
    model_info TEXT DEFAULT '{}',
    score_semantics TEXT DEFAULT '',
    status TEXT DEFAULT 'created',
    output_dir TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    finished_at TEXT,
    notes TEXT DEFAULT ''
);

-- AI / algorithm candidates. Never part of ground truth until reviewed & converted.
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    frame_id INTEGER NOT NULL REFERENCES frames(id),
    model_id TEXT NOT NULL,
    source_algorithm TEXT NOT NULL,
    class_name TEXT NOT NULL,
    score REAL NOT NULL,
    geometry_type TEXT NOT NULL,
    geometry TEXT NOT NULL,
    area_px REAL,
    inference_ms REAL,
    group_key TEXT,                           -- multi-view grouping (same physical candidate)
    review_status TEXT DEFAULT 'pending',     -- pending | accepted | rejected | unknown
    linked_annotation_id INTEGER,
    features TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_cand_run ON candidates(run_id);
CREATE INDEX IF NOT EXISTS ix_cand_frame ON candidates(frame_id);

CREATE TABLE IF NOT EXISTS splits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    seed INTEGER,
    config TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS split_assignments (
    split_id INTEGER NOT NULL REFERENCES splits(id),
    bottle_id INTEGER NOT NULL REFERENCES bottles(id),
    subset TEXT NOT NULL,                     -- train | val | test | excluded
    PRIMARY KEY (split_id, bottle_id)
);
"""


class Database:
    """Thin thread-aware wrapper around sqlite3.

    One connection per thread (sqlite3 objects are not shareable across threads by default).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()
        conn = self.conn
        conn.executescript(SCHEMA)
        cur = conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            conn.commit()
        elif int(row[0]) > SCHEMA_VERSION:
            raise RuntimeError(
                f"Project schema version {row[0]} is newer than this software ({SCHEMA_VERSION}). Update the software."
            )

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params=()):
        return self.conn.execute(sql, params).fetchone()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None
