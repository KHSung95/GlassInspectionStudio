"""File-based and mock providers (no hardware access)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..core.dataset_import import list_images
from ..core.image_io import read_image, to_mono
from .interface import CameraProvider, LightController, RotationStageProvider


class MockRotationStage(RotationStageProvider):
    def __init__(self):
        self._pos = 0.0

    def home(self):
        self._pos = 0.0

    def move_to(self, angle_deg):
        self._pos = float(angle_deg) % 360.0

    def position(self):
        return self._pos


class MockLightController(LightController):
    def __init__(self):
        self._s = {}

    def set_intensity(self, channel, value):
        self._s[int(channel)] = float(value)

    def state(self):
        return dict(self._s)


class MockCamera(CameraProvider):
    """Renders the synthetic bottle at the stage angle."""

    def __init__(self, stage: MockRotationStage, width=640, height=512, bit_depth=12, seed=0):
        from ..synthetic import _texture, default_bottles
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self.spec = default_bottles(1, width, height, seed)[0]
        self.spec.bit_depth = bit_depth
        self.tex, _ = _texture(self.spec, self.rng)
        self.exposure, self.gain, self.opened = 1000.0, 0.0, False

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def info(self):
        return {"camera_model": "MOCK-CAM", "bit_depth": self.spec.bit_depth, "exposure_us": self.exposure,
                "gain_db": self.gain}

    def set_exposure(self, us):
        self.exposure = float(us)

    def set_gain(self, db):
        self.gain = float(db)

    def grab(self, timeout_s=None):
        if not self.opened:
            raise RuntimeError("camera not opened")
        from ..synthetic import render
        return render(self.spec, self.stage.position(), self.rng, self.tex)


class FileCameraProvider(CameraProvider):
    """Replays an existing image folder as if it were a camera (angle taken from the stage by nearest frame)."""

    def __init__(self, folder: str | Path, stage: RotationStageProvider | None = None):
        self.files = list_images(Path(folder))
        if not self.files:
            raise FileNotFoundError(f"No images in {folder}")
        self.stage = stage
        self.i = 0

    def open(self):
        self.i = 0

    def close(self):
        pass

    def info(self):
        return {"camera_model": "FILE-REPLAY", "source": str(self.files[0].parent)}

    def set_exposure(self, us):
        pass

    def set_gain(self, db):
        pass

    def grab(self, timeout_s=None):
        if self.stage is not None:
            k = int(round(self.stage.position() / 360.0 * len(self.files))) % len(self.files)
        else:
            k = self.i % len(self.files)
            self.i += 1
        return to_mono(read_image(self.files[k]))
