"""Main window: dockable layout, shortcuts, viewer wiring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QScrollArea, QSplitter, QToolBar)

from ..core.geometry import pixel_index
from ..core.image_io import load_mono
from ..core.preprocessing import roi_stats
from ..inference.engine import candidates as load_candidates, load_maps
from .panels_annotation import AnnotationPanel, DisplayPanel, TimelinePanel
from .panels_dataset import (DatasetPanel, PropertiesPanel, run_import, show_duplicates, show_session_compare)
from .panels_lab import AlgorithmLab
from .panels_train_eval import EvaluationPanel, ReviewPanel, TrainingPanel
from .state import AppState
from .viewer import ImageViewer, colormap_qimage
from .widgets import error_box


def dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text = QColor(37, 37, 38), QColor(45, 45, 48), QColor(220, 220, 220)
    for role, c in ((QPalette.Window, alt), (QPalette.WindowText, text), (QPalette.Base, base),
                    (QPalette.AlternateBase, alt), (QPalette.ToolTipBase, base), (QPalette.ToolTipText, text),
                    (QPalette.Text, text), (QPalette.Button, alt), (QPalette.ButtonText, text),
                    (QPalette.BrightText, Qt.red), (QPalette.Highlight, QColor(38, 79, 120)),
                    (QPalette.HighlightedText, Qt.white), (QPalette.PlaceholderText, QColor(140, 140, 140))):
        p.setColor(role, c)
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(p)


class MainWindow(QMainWindow):
    STATE_KEY = "dock_state_v3"
    STATE_VERSION = 3

    def __init__(self, project: str | None = None):
        super().__init__()
        self.setWindowTitle("Glass Inspection Studio")
        self.resize(1700, 1000)
        self.st = AppState()
        self.settings = QSettings("GlassInspectionStudio", "GIS")
        self.show_cands = True
        self.show_anns = True
        self.preview_cands: list[dict] = []
        self.overlay_map = None
        # ---------------- central viewers
        self.viewer = ImageViewer()
        self.compare = ImageViewer(); self.compare.hide()
        sp = QSplitter(); sp.addWidget(self.viewer); sp.addWidget(self.compare)
        self.setCentralWidget(sp)
        self.viewer.viewChanged.connect(self._sync_compare)
        self.viewer.mouseMoved.connect(self._cursor)
        self.viewer.shapeCreated.connect(self._shape)
        self.viewer.itemClicked.connect(self._item_clicked)
        self.viewer.roiMeasured.connect(self._roi)
        # ---------------- panels
        self.dataset = DatasetPanel(self.st); self.props = PropertiesPanel(self.st)
        self.display = DisplayPanel(self.st); self.annot = AnnotationPanel(self.st)
        self.timeline = TimelinePanel(self.st)
        self.classical = AlgorithmLab(self.st, "classical"); self.zeroshot = AlgorithmLab(self.st, "zero_shot")
        self.supervised = AlgorithmLab(self.st, "supervised")
        self.training = TrainingPanel(self.st); self.evaluation = EvaluationPanel(self.st); self.review = ReviewPanel(self.st)
        from .panels_acquisition import AcquisitionPanel
        self.acq = AcquisitionPanel(self.st)
        self.acq.liveFrame.connect(self._show_live)
        self.evaluation.timeline = self.timeline
        self.review.center = lambda x, y: self.viewer.center_on_image(x, y, max(self.viewer.zoom(), 4.0))
        self.logw = QPlainTextEdit(); self.logw.setReadOnly(True); self.logw.setMaximumBlockCount(5000)
        docks = {}

        def dock(w, title, area):
            d = QDockWidget(title, self); d.setObjectName(title)
            if not isinstance(w, QPlainTextEdit):
                sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(w); sa.setFrameShape(QScrollArea.NoFrame)
                d.setWidget(sa)
            else:
                d.setWidget(w)
            d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
            self.addDockWidget(area, d); docks[title] = d
            return d
        dock(self.dataset, "Dataset Manager", Qt.LeftDockWidgetArea)
        r = [dock(self.annot, "Annotation Studio", Qt.RightDockWidgetArea),
             dock(self.display, "Display / Histogram", Qt.RightDockWidgetArea),
             dock(self.props, "Properties", Qt.RightDockWidgetArea),
             dock(self.classical, "Classical Vision Lab", Qt.RightDockWidgetArea),
             dock(self.zeroshot, "Zero-shot Inference Lab", Qt.RightDockWidgetArea),
             dock(self.supervised, "Model Inference", Qt.RightDockWidgetArea),
             dock(self.review, "Active Learning Review", Qt.RightDockWidgetArea),
             dock(self.acq, "Acquisition", Qt.RightDockWidgetArea)]
        for d in r[1:]:
            self.tabifyDockWidget(r[0], d)
        r[0].raise_()
        dock(self.timeline, "Rotation Timeline", Qt.BottomDockWidgetArea)
        b = [dock(self.training, "Training Lab", Qt.BottomDockWidgetArea),
             dock(self.evaluation, "Evaluation Lab", Qt.BottomDockWidgetArea),
             dock(self.logw, "Log / Tasks", Qt.BottomDockWidgetArea)]
        self.splitDockWidget(docks["Rotation Timeline"], b[0], Qt.Vertical)
        self.tabifyDockWidget(b[0], b[1]); self.tabifyDockWidget(b[0], b[2]); b[2].raise_()
        self.docks = docks
        self.resizeDocks([docks["Dataset Manager"], docks["Annotation Studio"]], [300, 440], Qt.Horizontal)
        self.resizeDocks([docks["Rotation Timeline"], docks["Training Lab"]], [160, 200], Qt.Vertical)
        for d in docks.values():
            d.setMinimumHeight(60)
        # ---------------- signals
        self.st.log.connect(self.log)
        self.st.frameChanged.connect(self.render_frame)
        self.st.annotationsChanged.connect(self.render_overlays)
        self.st.candidatesChanged.connect(self.render_overlays)
        self.st.sessionChanged.connect(self._session_scores)
        self.st.runsChanged.connect(self._session_scores)
        self.display.changed.connect(self.render_frame)
        self.annot.toolRequested.connect(self.set_tool)
        self.annot.brush.valueChanged.connect(lambda v: setattr(self.viewer, "brush_radius", v))
        for lab in (self.classical, self.zeroshot, self.supervised):
            lab.overlayRequested.connect(self._overlay)
            lab.previewCandidates.connect(self._preview)
            lab.parent_viewer_center = lambda x, y: self.viewer.center_on_image(x, y, max(self.viewer.zoom(), 4.0))
        self._menus(); self._toolbar(); self._shortcuts()
        self.status = QLabel(); self.statusBar().addWidget(self.status, 1)
        from ..core.device import detect
        dev = detect()
        self.devlabel = QLabel(f"torch {dev.get('torch')} | CUDA: {dev['cuda_available']} {dev['cuda_devices']} | "
                               f"nvidia-smi: {dev['nvidia_smi'] or 'not found'}")
        self.statusBar().addPermanentWidget(self.devlabel)
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        st = self.settings.value(self.STATE_KEY)
        if st is not None:
            self.restoreState(st, self.STATE_VERSION)
        last = project or self.settings.value("last_project")
        if last and Path(last, "studio.db").exists():
            self.open_project(str(last))

    # ------------------------------------------------------------------ menus / shortcuts
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        for t, fn, sc in (("New project…", self.new_project, "Ctrl+N"), ("Open project…", self.open_project_dialog, "Ctrl+O"),
                          ("Import image folder…", lambda: self._w(run_import(self, self.st, self.dataset.refresh)), "Ctrl+I"),
                          ("Find duplicate frames", lambda: show_duplicates(self, self.st), None),
                          ("Compare capture sessions", lambda: show_session_compare(self, self.st), None),
                          ("Export annotations (JSON)…", self.export_annotations, None),
                          ("Quit", self.close, "Ctrl+Q")):
            a = QAction(t, self); a.triggered.connect(fn)
            if sc:
                a.setShortcut(sc)
            m.addAction(a)
        e = self.menuBar().addMenu("&Edit")
        for t, fn, sc in (("Undo", self.undo, QKeySequence.Undo), ("Redo", self.redo, QKeySequence.Redo),
                          ("Search labels…", self.search_labels, "Ctrl+F")):
            a = QAction(t, self); a.triggered.connect(fn); a.setShortcut(sc); e.addAction(a)
        v = self.menuBar().addMenu("&View")
        for t, fn, sc in (("Fit (F)", self.viewer.fit, "F"), ("1:1 pixels", self.viewer.one_to_one, "Ctrl+1"),
                          ("Toggle candidates (C)", self._toggle_cands, "C"),
                          ("Toggle annotations (L)", self._toggle_anns, "L")):
            a = QAction(t, self); a.triggered.connect(fn); a.setShortcut(sc); v.addAction(a)
        v.addSeparator()
        for d in self.docks.values():
            v.addAction(d.toggleViewAction())
        h = self.menuBar().addMenu("&Help")
        a = QAction("Environment / GPU", self); a.triggered.connect(self._env); h.addAction(a)
        a = QAction("About", self); a.triggered.connect(lambda: QMessageBox.about(
            self, "Glass Inspection Studio", "Local machine-vision R&D platform for transparent glass bottle inspection.\n"
            "All data stays local. Zero-shot / model scores are uncalibrated candidates, not defect probabilities."))
        h.addAction(a)

    def _toolbar(self):
        tb = QToolBar("Main"); tb.setObjectName("main_toolbar"); self.addToolBar(tb)
        tb.addWidget(QLabel(" Compare: "))
        self.compare_mode = QComboBox(); self.compare_mode.addItems(["off", "previous frame", "next frame",
                                                                     "original (full range)", "score overlay only"])
        self.compare_mode.currentTextChanged.connect(lambda _: self.render_frame())
        tb.addWidget(self.compare_mode)
        tb.addSeparator()
        tb.addWidget(QLabel(" Show run: "))
        self.run_cb = QComboBox(); self.run_cb.setMinimumWidth(320); self.run_cb.currentIndexChanged.connect(self._run_selected)
        tb.addWidget(self.run_cb)
        self.st.runsChanged.connect(self._fill_runs); self.st.projectChanged.connect(self._fill_runs)

    def _shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self); s.setContext(Qt.WindowShortcut); s.activated.connect(fn)
        for t, k in AnnotationPanel.TOOL_KEYS:
            sc(k, lambda t=t: self.set_tool(t))
        sc(Qt.Key_Left, lambda: self.st.step(-1)); sc(Qt.Key_Right, lambda: self.st.step(1))
        sc(Qt.Key_Delete, self.annot.delete)
        sc("A", lambda: self.review.decide("accept")); sc("X", lambda: self.review.decide("reject"))
        sc("Shift+U", lambda: self.review.decide("unknown"))
        sc("Space", lambda: self.timeline.play.toggle())
        sc("F8", self.acq.capture)
        self._class_shortcuts = []

    def _bind_class_shortcuts(self):
        for s in self._class_shortcuts:
            s.setParent(None)
        self._class_shortcuts = []
        for c in self.st.project.classes():
            if c["shortcut"]:
                s = QShortcut(QKeySequence(c["shortcut"]), self)
                s.activated.connect(lambda n=c["name"]: self.annot.select_class(n))
                self._class_shortcuts.append(s)

    def _w(self, worker):
        self._worker = worker

    # ------------------------------------------------------------------ project
    def new_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose an EMPTY folder for the new project")
        if d:
            try:
                self.st.open_project(d, create=True); self._after_open(d)
            except Exception as e:
                error_box(self, "New project", repr(e))

    def open_project_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open project folder (contains studio.db)")
        if d:
            self.open_project(d)

    def open_project(self, d: str):
        try:
            self.st.open_project(d); self._after_open(d)
        except Exception as e:
            error_box(self, "Open project", repr(e))

    def _after_open(self, d):
        self.settings.setValue("last_project", d)
        self.setWindowTitle(f"Glass Inspection Studio — {self.st.project.name}")
        self._bind_class_shortcuts()
        s = self.st.project.stats()
        self.log(f"project stats: {s}")
        if s["ok_bottles"] == 0:
            self.log("NOTE: no verified OK bottles in this project — false-reject rate cannot be measured.")
        b = self.st.project.bottles()
        if b:
            ss = self.st.project.sessions(b[0]["id"])
            if ss:
                self.st.set_session(ss[0]["id"])

    def _show_live(self, img):
        bd = None
        try:
            bd = self.acq.cam.info().get("bit_depth") if self.acq.cam is not None and hasattr(self.acq.cam, "nodes") else None
        except Exception:
            pass
        dt = self.display.transform(bd)
        self.viewer.set_image(img, dt.apply(img), keep_view=True)
        self.viewer.clear_items("ann"); self.viewer.clear_items("cand"); self.viewer.set_overlay(None)
        self.status.setText(f"LIVE (not saved) {img.shape[1]}x{img.shape[0]} {img.dtype}")

    def closeEvent(self, e):
        self.acq.shutdown()
        self.settings.setValue(self.STATE_KEY, self.saveState(self.STATE_VERSION))
        super().closeEvent(e)

    # ------------------------------------------------------------------ rendering
    def log(self, msg: str):
        self.logw.appendPlainText(msg)

    def set_tool(self, t: str):
        self.viewer.set_tool(t)
        for k, b in self.annot.tool_btns.items():
            b.setChecked(k == t)
        self.status.setText(f"tool: {t}")

    def render_frame(self):
        f = self.st.frame
        if f is None:
            return
        try:
            raw = load_mono(f["path"])
        except Exception as e:
            error_box(self, "Cannot read image", f"{f['path']}\n{e!r}"); return
        dt = self.display.transform(f.get("bit_depth"))
        self.viewer.show_grid = self.display.grid.isChecked()
        self.viewer.set_image(raw, dt.apply(raw), keep_view=True)
        self.display.update_histogram(raw, f.get("bit_depth"), dt)
        # overlay: saved maps of selected run for this frame
        self.overlay_map = None
        rid = self.st.current_run_id
        if rid:
            maps = load_maps(self.st.project, rid, f["id"])
            if maps:
                self.overlay_map = (list(maps.values())[0], 0.5)
        self._apply_overlay()
        self.preview_cands = []
        self.render_overlays()
        self._render_compare(raw, dt)
        self.status.setText(f"{f['bottle_code']} / {f['session_code']}  angle {f['angle_deg']}°  "
                            f"{f['width']}x{f['height']} {f['dtype']}  review: {f['review_status']}")

    def _render_compare(self, raw, dt):
        mode = self.compare_mode.currentText()
        self.compare.setVisible(mode != "off")
        if mode == "off":
            return
        f = self.st.frame
        if mode in ("previous frame", "next frame"):
            g = self.st.frames[(self.st.index + (-1 if mode == "previous frame" else 1)) % len(self.st.frames)]
            r2 = load_mono(g["path"]); self.compare.set_image(r2, dt.apply(r2), keep_view=True)
            self.compare.show_annotations(self.st.store.for_frame(g["id"]), {c["name"]: c["color"] for c in
                                                                            self.st.project.classes(False)}, {})
        elif mode == "original (full range)":
            from ..core.preprocessing import DisplayTransform
            self.compare.set_image(raw, DisplayTransform(bit_depth=f.get("bit_depth")).apply(raw), keep_view=True)
        else:
            self.compare.set_image(raw, np.zeros(raw.shape, np.uint8), keep_view=True)
            if self.overlay_map is not None:
                m, _ = self.overlay_map
                self.compare.set_overlay(colormap_qimage(m, float(np.percentile(m, 50)), float(m.max()) + 1e-9, 1.0))
        self._sync_compare()

    def _sync_compare(self):
        if self.compare.isVisible():
            self.compare.set_view_state(self.viewer.view_state())

    def _apply_overlay(self):
        if self.overlay_map is None:
            self.viewer.set_overlay(None); return
        m, op = self.overlay_map
        lo = float(np.percentile(m, 99.0)); hi = float(m.max())
        self.viewer.set_overlay(colormap_qimage(m, lo, hi if hi > lo else lo + 1e-9), op)

    def _overlay(self, payload):
        self.overlay_map = payload
        self._apply_overlay()

    def _preview(self, cands):
        self.preview_cands = cands
        self.render_overlays()

    def render_overlays(self):
        f = self.st.frame
        if f is None:
            return
        colors = {c["name"]: c["color"] for c in self.st.project.classes(False)}
        if self.show_anns:
            anns = self.st.store.for_frame(f["id"])
            masks = {a["id"]: self.st.store.load_mask(a) for a in anns if a["geometry_type"] == "mask"}
            sel = {self.st.selected_annotation} if self.st.selected_annotation else set()
            self.viewer.show_annotations(anns, colors, masks, sel)
        else:
            self.viewer.clear_items("ann")
        cands = []
        if self.show_cands:
            if self.st.current_run_id:
                cands = load_candidates(self.st.project, self.st.current_run_id, f["id"])[:2000]
            cands = cands + self.preview_cands
        self.viewer.show_candidates(cands, self.st.selected_candidate)

    def _toggle_cands(self):
        self.show_cands = not self.show_cands; self.render_overlays()

    def _toggle_anns(self):
        self.show_anns = not self.show_anns; self.render_overlays()

    def _fill_runs(self):
        cur = self.st.current_run_id
        self.run_cb.blockSignals(True); self.run_cb.clear(); self.run_cb.addItem("(none)", None)
        for r in self.st.project.runs():
            if r["kind"] in ("classical", "zeroshot", "infer"):
                self.run_cb.addItem(f"#{r['id']} {r['algorithm']} {r['created_at']}", r["id"])
        i = self.run_cb.findData(cur)
        self.run_cb.setCurrentIndex(max(0, i)); self.run_cb.blockSignals(False)
        self.render_frame()

    def _run_selected(self):
        self.st.current_run_id = self.run_cb.currentData()
        self._session_scores(); self.render_frame()

    def _session_scores(self):
        """Per-angle max candidate score of the selected run → timeline graph."""
        if not self.st.frames:
            return
        rid = self.st.current_run_id
        if not rid:
            self.timeline.set_series({}); return
        xs, ys = [], []
        for i, f in enumerate(self.st.frames):
            r = self.st.project.db.one("SELECT MAX(score) FROM candidates WHERE run_id=? AND frame_id=?", (rid, f["id"]))
            xs.append(f["angle_deg"] if f["angle_deg"] is not None else i); ys.append(r[0] or 0.0)
        self.timeline.set_series({f"run #{rid} max score": (xs, ys)})

    # ------------------------------------------------------------------ interaction
    def _cursor(self, x, y):
        raw = self.viewer.raw
        if raw is None:
            return
        i, j = pixel_index(x, y)
        v = raw[j, i] if 0 <= j < raw.shape[0] and 0 <= i < raw.shape[1] else None
        txt = f"x: {x:.2f}  y: {y:.2f}  pixel: ({i}, {j})  value: {v if v is not None else '-'}  zoom: {self.viewer.zoom():.2f}x"
        if self.overlay_map is not None and v is not None:
            txt += f"  score: {self.overlay_map[0][j, i]:.4g}"
        self.display.cursor.setText(txt)

    def _roi(self, g):
        self.classical.roi = self.zeroshot.roi = self.supervised.roi = g
        s = roi_stats(self.viewer.raw, g["x0"], g["y0"], g["x1"], g["y1"])
        self.display.roi.setText("ROI stats: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                             for k, v in s.items()}))

    def _shape(self, tool: str, g: dict, mask):
        f = self.st.frame
        if f is None:
            return
        cmap = self.st.project.class_map()
        cls, state = self.st.current_class, self.st.current_state
        gtype = {"point": "point", "bbox": "bbox", "polygon": "polygon", "brush": "mask",
                 "normal_roi": "bbox", "ignore_roi": "bbox"}[tool]
        if tool == "normal_roi":
            state = "negative"
            cls = cls if cmap.get(cls, {}).get("kind") == "normal" else "Verified Normal"
        elif tool == "ignore_roi":
            state, cls = "ignore", "Unknown"
        try:
            self.st.store.add(f["id"], cls, state, gtype, g, mask=mask)
        except Exception as e:
            error_box(self, "Annotation", repr(e)); return
        self.st.annotationsChanged.emit()
        self.status.setText(f"saved {gtype} {cls} [{state}] (auto-saved)")

    def _item_clicked(self, kind, i):
        if kind == "annotation":
            self.st.selected_annotation = i
            for r in range(self.annot.table.rowCount()):
                if self.annot.table.item(r, 0).text() == str(i):
                    self.annot.table.selectRow(r)
            self.docks["Annotation Studio"].raise_()
        else:
            self.st.selected_candidate = i
            c = self.st.project.db.one("SELECT * FROM candidates WHERE id=?", (i,))
            self.log(f"candidate #{i}: {dict(c)}")
        self.render_overlays()

    def undo(self):
        if self.st.store and self.st.store.undo():
            self.st.annotationsChanged.emit()

    def redo(self):
        if self.st.store and self.st.store.redo():
            self.st.annotationsChanged.emit()

    def search_labels(self):
        from PySide6.QtWidgets import QInputDialog
        cls, ok = QInputDialog.getItem(self, "Search labels", "Class:", [""] + [c["name"] for c in self.st.project.classes()], 0, False)
        if not ok:
            return
        res = self.st.store.search(class_name=cls)
        items = [f"#{a['id']} f{a['frame_id']} {a['class_name']} {a['label_state']} angle={a['angle_deg']} "
                 f"{'confirmed' if a['confirmed'] else 'UNCONFIRMED'}" for a in res]
        if not items:
            QMessageBox.information(self, "Search", "No labels."); return
        it, ok = QInputDialog.getItem(self, "Search results", f"{len(items)} labels:", items, 0, False)
        if ok:
            a = res[items.index(it)]
            self.st.goto_frame_id(a["frame_id"])
            from ..core.geometry import geometry_center
            x, y = geometry_center(a["geometry_type"], a["geometry"])
            self.viewer.center_on_image(x, y, max(self.viewer.zoom(), 4.0))

    def export_annotations(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export annotations", str(self.st.project.root / "exports" / "annotations.json"), "*.json")
        if f:
            from ..core.export import export_project_annotations
            n = export_project_annotations(self.st.project, f)
            self.log(f"exported {n} annotations to {f}")

    def _env(self):
        from ..core.device import detect
        QMessageBox.information(self, "Environment", json.dumps(detect(), indent=1))


def main(argv=None) -> int:
    from ..algorithms.zero_shot.common import configure_model_cache
    configure_model_cache()
    app = QApplication(argv or sys.argv)
    app.setApplicationName("Glass Inspection Studio")
    dark_palette(app)
    proj = None
    args = (argv or sys.argv)[1:]
    if args:
        proj = args[0]
    w = MainWindow(proj)
    w.show()
    return app.exec()
