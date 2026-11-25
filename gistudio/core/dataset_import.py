"""Folder / CSV based import of rotation image sequences.

Supported layouts
-----------------
1. ``<root>/<bottle>/<session>/<frames>``  (layout='bottle/session')
2. ``<root>/<bottle>/<frames>``            (layout='bottle')   -> session code 'S001'
3. ``<root>/<frames>``                     (layout='flat')     -> bottle code from argument

Angle mapping priority: CSV (filename -> angle) > filename regex group 'angle' > frame_index * angle_step.
Files are hashed (sha256) and referenced by absolute path. Nothing is copied or modified.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .image_io import SUPPORTED_EXT, file_sha256, probe_image
from .project import Project

DEFAULT_ANGLE_REGEX = r"(?:^|[_\-])(?:a|ang|angle|deg)[_\-]?(?P<angle>\d+(?:\.\d+)?)|(?P<angle2>\d+(?:\.\d+)?)deg"
DEFAULT_INDEX_REGEX = r"(?P<index>\d+)(?!.*\d)"


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@dataclass
class ImportOptions:
    layout: str = "bottle/session"          # bottle/session | bottle | flat
    dataset_name: str = "default"
    bottle_code: str = ""                    # for flat layout
    angle_regex: str = DEFAULT_ANGLE_REGEX
    index_regex: str = DEFAULT_INDEX_REGEX
    angle_step: float | None = None          # fallback: angle = index_order * step
    angle_csv: str | None = None             # CSV with columns filename, angle[, frame_index]
    session_meta: dict = field(default_factory=dict)   # applied to every new session
    bottle_meta: dict = field(default_factory=dict)
    skip_duplicates: bool = True


@dataclass
class ImportReport:
    imported: int = 0
    skipped_existing: int = 0
    duplicates: list[tuple[str, str]] = field(default_factory=list)   # (new path, existing path)
    errors: list[tuple[str, str]] = field(default_factory=list)
    sessions: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (f"imported={self.imported} skipped_existing={self.skipped_existing} "
                f"duplicates={len(self.duplicates)} errors={len(self.errors)} sessions={len(self.sessions)}")


def list_images(d: Path) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT],
                  key=lambda p: natural_key(p.name))


def load_angle_csv(path: str) -> dict[str, tuple[float, int | None]]:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("filename") or row.get("file") or "").strip()
            if not name:
                continue
            ang = float(row["angle"]) if row.get("angle", "") != "" else None
            idx = int(row["frame_index"]) if row.get("frame_index", "") not in ("", None) else None
            out[Path(name).name] = (ang, idx)
    return out


def parse_angle(name: str, rx: re.Pattern) -> float | None:
    m = rx.search(Path(name).stem)
    if not m:
        return None
    g = m.groupdict()
    v = g.get("angle") or g.get("angle2")
    return float(v) if v is not None else None


def import_sequence(project: Project, session_id: int, files: list[Path], opts: ImportOptions,
                    report: ImportReport, progress=None) -> None:
    angle_rx = re.compile(opts.angle_regex, re.IGNORECASE) if opts.angle_regex else None
    index_rx = re.compile(opts.index_regex) if opts.index_regex else None
    csv_map = load_angle_csv(opts.angle_csv) if opts.angle_csv else {}
    db = project.db
    for order, f in enumerate(files):
        try:
            existing = db.one("SELECT id FROM frames WHERE session_id=? AND path=?", (session_id, str(f.resolve())))
            if existing:
                report.skipped_existing += 1
                continue
            sha = file_sha256(f)
            dup = db.one("SELECT path FROM frames WHERE sha256=?", (sha,))
            if dup:
                report.duplicates.append((str(f), dup["path"]))
                if opts.skip_duplicates:
                    continue
            info = probe_image(f)
            angle, idx = None, None
            if f.name in csv_map:
                angle, idx = csv_map[f.name]
            if angle is None and angle_rx is not None:
                angle = parse_angle(f.name, angle_rx)
            if idx is None and index_rx is not None:
                m = index_rx.search(f.stem)
                idx = int(m.group("index")) if m else None
            if idx is None:
                idx = order
            if angle is None and opts.angle_step:
                angle = order * float(opts.angle_step)
            db.execute(
                """INSERT INTO frames(session_id, frame_index, angle_deg, path, sha256, file_size, width, height, dtype,
                   container_bits) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (session_id, idx, angle, str(f.resolve()), sha, f.stat().st_size, info.width, info.height,
                 info.dtype, info.container_bits),
            )
            report.imported += 1
        except Exception as e:  # keep importing other files
            report.errors.append((str(f), repr(e)))
        if progress:
            progress(order + 1, len(files), f.name)
    db.commit()


def import_folder(project: Project, root: str | Path, opts: ImportOptions, progress=None) -> ImportReport:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    rep = ImportReport()
    ds = project.ensure_dataset(opts.dataset_name)

    def do(bottle_code: str, session_code: str, d: Path) -> None:
        files = list_images(d)
        if not files:
            return
        b = project.ensure_bottle(ds, bottle_code, **opts.bottle_meta)
        smeta = dict(opts.session_meta)
        meta_file = d / "session_meta.json"
        if meta_file.exists():
            smeta.update(json.loads(meta_file.read_text(encoding="utf-8")))
            smeta = {k: v for k, v in smeta.items() if k in Project._SESSION_FIELDS}
        smeta.setdefault("source_dir", str(d.resolve()))
        s = project.ensure_session(b, session_code, **smeta)
        rep.sessions.append(s)
        local = ImportOptions(**{**opts.__dict__})
        if local.angle_csv is None and (d / "angles.csv").exists():
            local.angle_csv = str(d / "angles.csv")
        import_sequence(project, s, files, local, rep, progress)

    if opts.layout == "flat":
        do(opts.bottle_code or root.name, "S001", root)
    elif opts.layout == "bottle":
        for bd in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
            do(bd.name, "S001", bd)
    elif opts.layout == "bottle/session":
        for bd in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
            subs = sorted([p for p in bd.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))
            if not subs:  # tolerate bottle folders without sessions
                do(bd.name, "S001", bd)
            for sd in subs:
                do(bd.name, sd.name, sd)
    else:
        raise ValueError(opts.layout)
    return rep


def find_duplicates(project: Project) -> list[list[str]]:
    rows = project.db.query(
        "SELECT sha256, GROUP_CONCAT(path, '|') AS paths, COUNT(*) AS n FROM frames GROUP BY sha256 HAVING n>1")
    return [r["paths"].split("|") for r in rows]


def compare_sessions(project: Project, session_a: int, session_b: int) -> dict:
    """Compare two capture sessions (e.g. repeated captures of the same bottle)."""
    fa, fb = project.frames(session_a), project.frames(session_b)
    sa, sb = project.session(session_a), project.session(session_b)
    keys = ["camera_model", "bit_depth", "exposure_us", "gain_db", "lighting", "lens", "magnification", "mm_per_px"]
    angles_a = sorted(a["angle_deg"] for a in fa if a["angle_deg"] is not None)
    angles_b = sorted(b["angle_deg"] for b in fb if b["angle_deg"] is not None)
    return {
        "frames": (len(fa), len(fb)),
        "resolution": ({(f["width"], f["height"]) for f in fa}, {(f["width"], f["height"]) for f in fb}),
        "angle_range": ((angles_a[0], angles_a[-1]) if angles_a else None, (angles_b[0], angles_b[-1]) if angles_b else None),
        "meta_diff": {k: (sa.get(k), sb.get(k)) for k in keys if sa.get(k) != sb.get(k)},
    }
