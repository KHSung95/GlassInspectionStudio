"""Abstract hardware interfaces. Real SDK adapters (GenICam/GigE Vision/USB3 Vision/Spinnaker, motion
controllers) should subclass these in separate modules that are imported only when the SDK is installed."""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameMeta:
    index: int
    angle_deg: float | None
    exposure_us: float | None = None
    gain_db: float | None = None
    timestamp: str = ""


class CameraProvider(abc.ABC):
    @abc.abstractmethod
    def open(self) -> None: ...
    @abc.abstractmethod
    def close(self) -> None: ...
    @abc.abstractmethod
    def info(self) -> dict: ...
    @abc.abstractmethod
    def set_exposure(self, us: float) -> None: ...
    @abc.abstractmethod
    def set_gain(self, db: float) -> None: ...
    @abc.abstractmethod
    def grab(self) -> np.ndarray: ...


class RotationStageProvider(abc.ABC):
    @abc.abstractmethod
    def home(self) -> None: ...
    @abc.abstractmethod
    def move_to(self, angle_deg: float) -> None: ...
    @abc.abstractmethod
    def position(self) -> float: ...


class LightController(abc.ABC):
    @abc.abstractmethod
    def set_intensity(self, channel: int, value: float) -> None: ...
    @abc.abstractmethod
    def state(self) -> dict: ...


class CaptureCoordinator:
    """Step-and-shoot acquisition of one rotation sequence into a session folder (lossless TIFF + angles.csv)."""

    def __init__(self, camera: CameraProvider, stage: RotationStageProvider, light: LightController | None = None):
        self.camera, self.stage, self.light = camera, stage, light

    def capture_sequence(self, out_dir: str | Path, step_deg: float, start_deg: float = 0.0,
                         n_frames: int | None = None, progress=None) -> Path:
        import csv
        from datetime import datetime
        import tifffile
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise FileExistsError(f"{out} is not empty (refusing to overwrite captured data)")
        n = n_frames or int(round(360.0 / step_deg))
        self.camera.open()
        try:
            self.stage.home()
            rows = []
            for i in range(n):
                a = (start_deg + i * step_deg) % 360.0
                self.stage.move_to(a)
                img = self.camera.grab()
                name = f"frame_{i:04d}.tif"
                tifffile.imwrite(out / name, img)
                rows.append([name, f"{self.stage.position():.4f}", i, datetime.now().isoformat()])
                if progress:
                    progress(i + 1, n, name)
            with open(out / "angles.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["filename", "angle", "frame_index", "timestamp"]); w.writerows(rows)
            (out / "session_meta.json").write_text(json.dumps({**self.camera.info(),
                                                               **({"lighting": json.dumps(self.light.state())}
                                                                  if self.light else {})}), encoding="utf-8")
        finally:
            self.camera.close()
        return out
