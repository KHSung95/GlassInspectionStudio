"""Operator-driven capture into a project (manual rotation now; stage/encoder handshake later).

Every frame is written losslessly (TIFF, native bit container) into
<project>/acquired/<bottle>/<session>/frame_XXXX.tif, appended to angles.csv, and registered in the project
database immediately (crash-safe: what is on disk is also in the DB). Files are never overwritten.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from ..core.image_io import file_sha256
from ..core.project import Project


@dataclass
class CaptureTarget:
    bottle_code: str
    session_code: str = ""
    product_type: str = ""
    lot_no: str = ""
    memo: str = ""
    dataset: str = "acquired"


class CaptureSessionWriter:
    def __init__(self, project: Project, target: CaptureTarget, camera_info: dict):
        self.p = project
        self.t = target
        if not target.bottle_code.strip():
            raise ValueError("bottle code is required")
        ds = project.ensure_dataset(target.dataset)
        bfields = {k: v for k, v in (("product_type", target.product_type), ("lot_no", target.lot_no)) if v}
        self.bottle_id = project.ensure_bottle(ds, target.bottle_code.strip(), **bfields)
        code = target.session_code.strip() or datetime.now().strftime("S%Y%m%d-%H%M%S")
        self.dir = project.root / "acquired" / target.bottle_code.strip() / code
        if self.dir.exists() and any(self.dir.glob("frame_*.tif")):
            raise FileExistsError(f"Session folder already has frames: {self.dir} (choose a new session code)")
        self.dir.mkdir(parents=True, exist_ok=True)
        smeta = {"camera_model": camera_info.get("camera_model", ""), "bit_depth": camera_info.get("bit_depth"),
                 "exposure_us": camera_info.get("exposure_us"), "gain_db": camera_info.get("gain_db"),
                 "captured_at": datetime.now().isoformat(timespec="seconds"), "source_dir": str(self.dir),
                 "memo": (f"serial={camera_info.get('camera_serial')} pixel_format={camera_info.get('pixel_format')} "
                          + target.memo).strip()}
        self.session_id = project.ensure_session(self.bottle_id, code, **{k: v for k, v in smeta.items() if v is not None})
        self.session_code = code
        (self.dir / "session_meta.json").write_text(json.dumps(smeta, indent=1, default=str), encoding="utf-8")
        if camera_info.get("features"):
            (self.dir / "camera_features.json").write_text(json.dumps(camera_info["features"], indent=1, default=str),
                                                           encoding="utf-8")
        self.count = 0
        csv_path = self.dir / "angles.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["filename", "angle", "frame_index", "timestamp", "exposure_us", "gain_db"])

    def add(self, img: np.ndarray, angle_deg: float | None, exposure_us=None, gain_db=None) -> int:
        """Persist one frame and register it. Returns the frame id."""
        if img.ndim != 2:
            raise ValueError("mono image expected")
        i = self.count
        name = f"frame_{i:04d}.tif"
        path = self.dir / name
        if path.exists():
            raise FileExistsError(path)
        tifffile.imwrite(path, img)
        ts = datetime.now().isoformat(timespec="milliseconds")
        with open(self.dir / "angles.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, "" if angle_deg is None else f"{angle_deg:.3f}", i, ts, exposure_us, gain_db])
        cur = self.p.db.execute(
            """INSERT INTO frames(session_id, frame_index, angle_deg, path, sha256, file_size, width, height, dtype,
               container_bits, captured_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session_id, i, angle_deg, str(path.resolve()), file_sha256(path), path.stat().st_size,
             int(img.shape[1]), int(img.shape[0]), str(img.dtype), img.dtype.itemsize * 8, ts))
        self.p.db.commit()
        self.count += 1
        return cur.lastrowid
