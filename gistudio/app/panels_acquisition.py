"""Acquisition panel: connect a GenICam camera (or the mock), live view, manual-rotation capture into the project.

Camera features are only written when the user presses an explicit 'Apply' button.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..acquisition.capture_session import CaptureSessionWriter, CaptureTarget
from .state import AppState
from .widgets import Worker, error_box


def image_quality(img: np.ndarray, bit_depth: int | None) -> dict:
    """Quick acquisition QA: focus (variance of Laplacian on a central crop), saturation and dark fraction."""
    h, w = img.shape
    c = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4].astype(np.float32)
    step = max(1, c.shape[0] // 1024)
    c = c[::step, ::step]
    full = float((1 << bit_depth) - 1) if bit_depth else float(np.iinfo(img.dtype).max if img.dtype.kind == "u" else 1)
    sub = img[::4, ::4]
    return {"focus_lapvar": float(cv2.Laplacian(c / full, cv2.CV_32F).var() * 1e6),
            "saturated_pct": float((sub >= full).mean() * 100), "dark_pct": float((sub <= full * 0.02).mean() * 100),
            "mean": float(sub.mean()), "max": float(sub.max())}


class LiveThread(QThread):
    frame = Signal(object)
    failed = Signal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._run = True

    def stop(self):
        self._run = False

    def run(self):
        while self._run:
            try:
                self.frame.emit(self.camera.grab(timeout_s=5.0))
            except Exception as e:
                self.failed.emit(repr(e)); return
            time.sleep(0.05)


class AcquisitionPanel(QWidget):
    liveFrame = Signal(object)          # np.ndarray (not saved)
    captured = Signal(int)              # frame id

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        self.cam = None
        self.live: LiveThread | None = None
        self.writer: CaptureSessionWriter | None = None
        self.worker = None
        self.next_angle = 0.0
        lay = QVBoxLayout(self)
        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        self.kind = QComboBox(); self.kind.addItems(["GenICam (Spinnaker GenTL)", "Mock (synthetic bottle)"])
        from ..acquisition.genicam_camera import DEFAULT_CTI
        self.cti = QLineEdit(DEFAULT_CTI)
        row = QHBoxLayout(); self.bconn = QPushButton("Connect"); self.bconn.clicked.connect(self.connect_camera)
        self.bdisc = QPushButton("Disconnect"); self.bdisc.clicked.connect(self.disconnect_camera)
        row.addWidget(self.bconn); row.addWidget(self.bdisc)
        cf.addRow("Source", self.kind); cf.addRow("GenTL .cti", self.cti); cf.addRow(row)
        self.info = QLabel("not connected — close SpinView first (one application per camera)"); self.info.setWordWrap(True)
        cf.addRow(self.info)
        self.pf = QComboBox(); bpf = QPushButton("Apply"); bpf.clicked.connect(self._apply_pf)
        r = QHBoxLayout(); r.addWidget(self.pf, 1); r.addWidget(bpf); cf.addRow("Pixel format", r)
        self.exp = QDoubleSpinBox(); self.exp.setRange(1, 10_000_000); self.exp.setDecimals(1); self.exp.setSuffix(" µs")
        bexp = QPushButton("Apply"); bexp.clicked.connect(lambda: self._apply("set_exposure", self.exp.value()))
        r = QHBoxLayout(); r.addWidget(self.exp, 1); r.addWidget(bexp); cf.addRow("Exposure", r)
        self.gain = QDoubleSpinBox(); self.gain.setRange(-10, 48); self.gain.setDecimals(2); self.gain.setSuffix(" dB")
        bg = QPushButton("Apply"); bg.clicked.connect(lambda: self._apply("set_gain", self.gain.value()))
        r = QHBoxLayout(); r.addWidget(self.gain, 1); r.addWidget(bg); cf.addRow("Gain", r)
        lay.addWidget(cg)
        lg = QGroupBox("Live / QA"); ll = QVBoxLayout(lg)
        self.blive = QPushButton("Live view (not saved)"); self.blive.setCheckable(True); self.blive.toggled.connect(self.toggle_live)
        ll.addWidget(self.blive)
        self.qa = QLabel("-"); self.qa.setWordWrap(True); ll.addWidget(self.qa)
        lay.addWidget(lg)
        sg = QGroupBox("Capture session (manual rotation)"); sf = QFormLayout(sg)
        self.bottle = QLineEdit(); self.bottle.setPlaceholderText("e.g. NG-001")
        self.ptype = QLineEdit(); self.lot = QLineEdit(); self.scode = QLineEdit(); self.scode.setPlaceholderText("auto: S<date-time>")
        self.memo = QLineEdit()
        self.start = QDoubleSpinBox(); self.start.setRange(-360, 360); self.start.setSuffix("°")
        self.step = QDoubleSpinBox(); self.step.setRange(0, 360); self.step.setValue(10); self.step.setSuffix("°")
        self.step.setSpecialValueText("no angle (static)")
        self.avg = QSpinBox(); self.avg.setRange(1, 64); self.avg.setValue(1)
        self.avg.setToolTip("Temporal averaging of N frames for static scenes (stored as uint16 if >8 bit)")
        self.burst = QSpinBox(); self.burst.setRange(1, 1000); self.burst.setValue(1)
        for n, w in (("Bottle code *", self.bottle), ("Product type", self.ptype), ("Lot", self.lot),
                     ("Session code", self.scode), ("Memo", self.memo), ("Start angle", self.start),
                     ("Angle step / shot", self.step), ("Average frames", self.avg), ("Shots per click", self.burst)):
            sf.addRow(n, w)
        r = QHBoxLayout(); bnew = QPushButton("New session"); bnew.clicked.connect(self.new_session)
        self.bcap = QPushButton("Capture (F8)"); self.bcap.clicked.connect(self.capture)
        bend = QPushButton("End session"); bend.clicked.connect(self.end_session)
        r.addWidget(bnew); r.addWidget(self.bcap); r.addWidget(bend); sf.addRow(r)
        self.sinfo = QLabel("no session"); self.sinfo.setWordWrap(True); sf.addRow(self.sinfo)
        lay.addWidget(sg)
        lay.addStretch(1)
        self._last_frame = None

    # ------------------------------------------------------------------ camera
    def connect_camera(self):
        if self.cam is not None:
            return
        try:
            if self.kind.currentIndex() == 0:
                from ..acquisition.genicam_camera import GenICamCamera
                cam = GenICamCamera(self.cti.text())
            else:
                from ..acquisition.mock import MockCamera, MockRotationStage
                cam = MockCamera(MockRotationStage(), width=1024, height=768)
                cam._stage_mock = True
            cam.open()
            self.cam = cam
            self._refresh_info()
            self.st.log.emit(f"camera connected: {self.info.text()}")
        except Exception as e:
            msg = repr(e)
            if "AccessDenied" in msg or "not allowed" in msg:
                msg += "\n\nThe camera is probably open in another application (SpinView). Close it and retry."
            error_box(self, "Camera", msg)

    def _refresh_info(self):
        i = self.cam.info()
        f = i.get("features", {})
        self.info.setText(f"{i.get('camera_model')}  S/N {i.get('camera_serial')}  {f.get('Width')}×{f.get('Height')}  "
                          f"{i.get('pixel_format')}  exp {i.get('exposure_us')} µs  gain {i.get('gain_db')} dB")
        if hasattr(self.cam, "available_pixel_formats"):
            self.pf.clear(); self.pf.addItems(self.cam.available_pixel_formats())
            if i.get("pixel_format"):
                self.pf.setCurrentText(i["pixel_format"])
        if i.get("exposure_us"):
            self.exp.setValue(float(i["exposure_us"]))
        if i.get("gain_db") is not None:
            self.gain.setValue(float(i["gain_db"]))

    def disconnect_camera(self):
        self.blive.setChecked(False)
        if self.cam is not None:
            try:
                self.cam.close()
            finally:
                self.cam = None
                self.info.setText("not connected")

    def _with_live_paused(self, fn):
        was = self.blive.isChecked()
        if was:
            self.blive.setChecked(False)
        try:
            return fn()
        finally:
            if was:
                self.blive.setChecked(True)

    def _apply(self, method, value):
        if self.cam is None:
            return
        try:
            self._with_live_paused(lambda: (getattr(self.cam, "stop", lambda: None)(), getattr(self.cam, method)(value)))
            self._refresh_info()
            self.st.log.emit(f"camera {method}({value}) applied by operator")
        except Exception as e:
            error_box(self, "Camera setting", repr(e))

    def _apply_pf(self):
        if self.cam is not None and hasattr(self.cam, "set_pixel_format") and self.pf.currentText():
            self._apply("set_pixel_format", self.pf.currentText())

    # ------------------------------------------------------------------ live
    def toggle_live(self, on):
        if on:
            if self.cam is None:
                self.blive.setChecked(False); return
            self.live = LiveThread(self.cam, self)
            self.live.frame.connect(self._on_live)
            self.live.failed.connect(lambda m: (self.blive.setChecked(False), error_box(self, "Live", m)))
            self.live.start()
        elif self.live is not None:
            self.live.stop(); self.live.wait(6000); self.live = None

    def _on_live(self, img):
        self._last_frame = img
        bd = self.cam.info().get("bit_depth") if hasattr(self.cam, "nodes") else 12
        q = image_quality(img, bd)
        self.qa.setText(f"focus {q['focus_lapvar']:.1f}  saturated {q['saturated_pct']:.2f}%  dark {q['dark_pct']:.1f}%  "
                        f"mean {q['mean']:.0f}  max {q['max']:.0f}  ({img.shape[1]}×{img.shape[0]} {img.dtype})")
        self.liveFrame.emit(img)

    # ------------------------------------------------------------------ capture
    def new_session(self):
        if self.st.project is None or self.cam is None:
            QMessageBox.information(self, "Capture", "Open a project and connect a camera first."); return
        try:
            t = CaptureTarget(self.bottle.text(), self.scode.text(), self.ptype.text(), self.lot.text(), self.memo.text())
            self.writer = CaptureSessionWriter(self.st.project, t, self._with_live_paused(self.cam.info))
            self.next_angle = self.start.value()
            self.sinfo.setText(f"session {self.writer.session_code} → {self.writer.dir}")
            self.st.projectChanged.emit()
        except Exception as e:
            error_box(self, "New session", repr(e))

    def capture(self):
        if self.writer is None:
            self.new_session()
            if self.writer is None:
                return
        if self.worker is not None and self.worker.isRunning():
            return
        n, avg = self.burst.value(), self.avg.value()
        step = self.step.value()
        live_was = self.blive.isChecked()
        self.blive.setChecked(False)
        cam, writer = self.cam, self.writer
        angles = []
        for k in range(n):
            angles.append(None if step == 0 else self.next_angle + k * step)

        def job(progress, stop):
            ids = []
            info = cam.info()
            for k, a in enumerate(angles):
                if stop():
                    break
                img = cam.grab_averaged(avg) if hasattr(cam, "grab_averaged") else cam.grab()
                ids.append(writer.add(img, a, info.get("exposure_us"), info.get("gain_db")))
                progress(k + 1, len(angles), f"frame {ids[-1]} angle {a}")
            return ids

        self.worker = Worker(job, self)

        def done(ids):
            if step:
                self.next_angle += step * len(ids)
            self.sinfo.setText(f"session {writer.session_code}: {writer.count} frames; next angle "
                               f"{self.next_angle if step else '-'}  → {writer.dir}")
            if ids:
                self.st.goto_frame_id(ids[-1])
                self.captured.emit(ids[-1])
            if live_was:
                self.blive.setChecked(True)
        self.worker.done.connect(done)
        self.worker.failed.connect(lambda tb: error_box(self, "Capture failed", tb))
        self.worker.start()

    def end_session(self):
        if self.writer is not None:
            self.st.log.emit(f"capture session ended: {self.writer.count} frames in {self.writer.dir}")
        self.writer = None
        self.sinfo.setText("no session")
        self.st.projectChanged.emit()

    def shutdown(self):
        self.disconnect_camera()
