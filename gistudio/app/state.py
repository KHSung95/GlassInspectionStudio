"""Application state shared by all panels (single source of truth + Qt signals)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..core.annotations import AnnotationStore
from ..core.geometry import circ_dist, wrap_deg
from ..core.image_io import load_mono
from ..core.project import Project


class AppState(QObject):
    projectChanged = Signal()
    sessionChanged = Signal()
    frameChanged = Signal()
    annotationsChanged = Signal()
    candidatesChanged = Signal()
    runsChanged = Signal()
    classChanged = Signal(str)
    log = Signal(str)

    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.store: AnnotationStore | None = None
        self.bottle_id: int | None = None
        self.session_id: int | None = None
        self.frames: list[dict] = []
        self.index: int = 0
        self.current_class = "Black Spot"
        self.current_state = "positive"
        self.current_run_id: int | None = None
        self.selected_annotation: int | None = None
        self.selected_candidate: int | None = None
        self.author = ""

    # ------------------------------------------------------------------ project
    def open_project(self, path: str, create: bool = False) -> None:
        self.project = Project.create(path) if create else Project(path)
        self.store = AnnotationStore(self.project, self.author)
        self.bottle_id = self.session_id = None
        self.frames, self.index = [], 0
        self.projectChanged.emit()
        self.log.emit(f"Project opened: {self.project.root}")

    # ------------------------------------------------------------------ navigation
    def set_session(self, session_id: int) -> None:
        self.session_id = session_id
        s = self.project.session(session_id)
        self.bottle_id = s["bottle_id"]
        self.frames = self.project.frames(session_id)
        self.index = 0
        self.sessionChanged.emit()
        if self.frames:
            self.frameChanged.emit()

    @property
    def frame(self) -> dict | None:
        if not self.frames:
            return None
        return self.project.frame(self.frames[self.index]["id"])

    def image(self) -> np.ndarray | None:
        f = self.frame
        return load_mono(f["path"]) if f else None

    def goto_index(self, i: int) -> None:
        if not self.frames:
            return
        self.index = i % len(self.frames)  # circular 0/360 wrap
        self.frameChanged.emit()

    def step(self, d: int) -> None:
        self.goto_index(self.index + d)

    def goto_angle(self, angle: float) -> None:
        if not self.frames:
            return
        a = wrap_deg(angle)
        best = min(range(len(self.frames)),
                   key=lambda i: circ_dist(self.frames[i]["angle_deg"] if self.frames[i]["angle_deg"] is not None
                                           else i, a))
        self.goto_index(best)

    def goto_frame_id(self, frame_id: int) -> None:
        f = self.project.frame(frame_id)
        if f["session_id"] != self.session_id:
            self.set_session(f["session_id"])
        for i, fr in enumerate(self.frames):
            if fr["id"] == frame_id:
                self.goto_index(i); return
