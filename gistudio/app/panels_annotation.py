"""Annotation Studio panels: display controls, annotation editor, rotation timeline."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..core.preprocessing import DisplayTransform, histogram
from .state import AppState


class DisplayPanel(QWidget):
    """Visualisation-only transforms (never used as model input)."""
    changed = Signal()

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.auto = QComboBox(); self.auto.addItems(["Full range", "Stretch 0.5-99.5%", "Window/Level"])
        self.level = QDoubleSpinBox(); self.level.setRange(-1e6, 1e6); self.level.setDecimals(1)
        self.window = QDoubleSpinBox(); self.window.setRange(1, 1e6); self.window.setDecimals(1)
        self.gamma = QDoubleSpinBox(); self.gamma.setRange(0.1, 5); self.gamma.setSingleStep(0.1); self.gamma.setValue(1.0)
        self.clahe = QCheckBox("CLAHE"); self.invert = QCheckBox("Invert")
        self.grid = QCheckBox("Pixel grid / values at high zoom"); self.grid.setChecked(True)
        for w, n in ((self.auto, "Mode"), (self.level, "Level"), (self.window, "Window"), (self.gamma, "Gamma")):
            f.addRow(n, w)
        f.addRow(self.clahe); f.addRow(self.invert); f.addRow(self.grid)
        lay.addLayout(f)
        for w in (self.auto,):
            w.currentIndexChanged.connect(self.changed)
        for w in (self.level, self.window, self.gamma):
            w.valueChanged.connect(self._wl_changed)
        for w in (self.clahe, self.invert, self.grid):
            w.toggled.connect(self.changed)
        self.hist = pg.PlotWidget(); self.hist.setMaximumHeight(140); self.hist.setBackground("#202020")
        self.hist.hideAxis("left")
        lay.addWidget(self.hist)
        self.cursor = QLabel("x: - y: - value: -"); lay.addWidget(self.cursor)
        self.roi = QLabel("ROI stats: (use Measure tool, key M)"); self.roi.setWordWrap(True)
        self.roi.setTextInteractionFlags(Qt.TextSelectableByMouse); lay.addWidget(self.roi)
        lay.addStretch(1)

    def _wl_changed(self):
        if self.sender() in (self.level, self.window):
            self.auto.blockSignals(True); self.auto.setCurrentIndex(2); self.auto.blockSignals(False)
        self.changed.emit()

    def transform(self, bit_depth: int | None) -> DisplayTransform:
        m = self.auto.currentIndex()
        return DisplayTransform(window=self.window.value() if m == 2 else None,
                                level=self.level.value() if m == 2 else None, gamma=self.gamma.value(),
                                invert=self.invert.isChecked(), stretch_percentiles=(0.5, 99.5) if m == 1 else None,
                                clahe=self.clahe.isChecked(), bit_depth=bit_depth)

    def update_histogram(self, raw: np.ndarray, bit_depth: int | None, dt: DisplayTransform):
        h, e = histogram(raw, 256, bit_depth)
        self.hist.clear()
        self.hist.plot(e, np.log1p(h), stepMode="center", fillLevel=0, brush=(120, 160, 255, 120))
        lo, hi = dt.value_range(raw)
        self.hist.addLine(x=lo, pen="y"); self.hist.addLine(x=hi, pen="y")
        if self.auto.currentIndex() != 2:
            for w, v in ((self.level, (lo + hi) / 2), (self.window, hi - lo)):
                w.blockSignals(True); w.setValue(v); w.blockSignals(False)


class AnnotationPanel(QWidget):
    toolRequested = Signal(str)

    TOOL_KEYS = [("select", "V"), ("pan", "H"), ("point", "P"), ("bbox", "B"), ("polygon", "G"), ("brush", "R"),
                 ("normal_roi", "O"), ("ignore_roi", "I"), ("measure", "M")]

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        tg = QGroupBox("Tools"); tl = QGridLayout(tg)
        self.tool_btns = {}
        for i, (t, k) in enumerate(self.TOOL_KEYS):
            b = QPushButton(f"{t} ({k})"); b.setCheckable(True); b.clicked.connect(lambda _=False, t=t: self.toolRequested.emit(t))
            tl.addWidget(b, i // 3, i % 3); self.tool_btns[t] = b
        self.brush = QSpinBox(); self.brush.setRange(1, 50); self.brush.setValue(3); tl.addWidget(QLabel("brush r"), 3, 0)
        tl.addWidget(self.brush, 3, 1)
        lay.addWidget(tg)
        sg = QGroupBox("Label state"); sl = QHBoxLayout(sg); self.state_grp = QButtonGroup(self)
        for i, s in enumerate(("positive", "negative", "unknown", "ignore")):
            r = QRadioButton(s); r.setChecked(i == 0); self.state_grp.addButton(r, i); sl.addWidget(r)
            r.toggled.connect(lambda on, s=s: on and setattr(self.st, "current_state", s))
        lay.addWidget(sg)
        self.classes = QListWidget(); self.classes.setMaximumHeight(170)
        self.classes.currentItemChanged.connect(self._class_selected)
        lay.addWidget(QLabel("Class (shortcut keys shown)")); lay.addWidget(self.classes)
        cb = QHBoxLayout(); addc = QPushButton("Add / edit class…"); addc.clicked.connect(self._edit_class); cb.addWidget(addc)
        lay.addLayout(cb)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["id", "class", "state", "type", "visibility", "confirmed", "defect", "source"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self._sel)
        lay.addWidget(self.table, 2)
        g = QGridLayout()
        btns = [("Delete (Del)", self.delete), ("Confirm", self.confirm), ("Keyframe", self.keyframe),
                ("New defect ID", self.new_defect), ("Link to defect…", self.link_defect),
                ("Copy → next N", self.propagate), ("Interpolate 2 sel.", self.interpolate),
                ("Track (template)", self.track)]
        for i, (t, fn) in enumerate(btns):
            b = QPushButton(t); b.clicked.connect(fn); g.addWidget(b, i // 2, i % 2)
        self.vis = QComboBox(); self.vis.addItems(["visible", "weak", "not_visible", "uncertain"])
        sv = QPushButton("Set visibility"); sv.clicked.connect(self.set_visibility)
        g.addWidget(self.vis, 4, 0); g.addWidget(sv, 4, 1)
        self.verdict = QComboBox(); self.verdict.addItems(["undecided", "ng", "ok"])
        svd = QPushButton("Set spec verdict"); svd.clicked.connect(self.set_verdict)
        g.addWidget(self.verdict, 5, 0); g.addWidget(svd, 5, 1)
        lay.addLayout(g)
        rg = QGroupBox("Frame review"); rl = QVBoxLayout(rg)
        self.review = QComboBox(); self.review.addItems(["unreviewed", "in_progress", "reviewed"])
        self.review.currentTextChanged.connect(self._review_status)
        rl.addWidget(self.review)
        rl.addWidget(QLabel("Exhaustively reviewed for (unlabelled = negative for that class only):"))
        self.scopes = QListWidget(); self.scopes.setMaximumHeight(90); self.scopes.itemChanged.connect(self._scope)
        rl.addWidget(self.scopes)
        self.overlap = QLabel(""); self.overlap.setWordWrap(True); rl.addWidget(self.overlap)
        lay.addWidget(rg)
        st.projectChanged.connect(self.load_classes)
        st.frameChanged.connect(self.refresh); st.annotationsChanged.connect(self.refresh)

    # ------------------------------------------------------------------ classes
    def load_classes(self):
        self.classes.clear()
        for c in self.st.project.classes():
            it = QListWidgetItem(f"[{c['shortcut']}] {c['name']}  ({c['kind']})")
            it.setData(Qt.UserRole, c["name"]); it.setForeground(pg.mkColor(c["color"]))
            self.classes.addItem(it)
        self.select_class(self.st.current_class)

    def select_class(self, name: str):
        for i in range(self.classes.count()):
            if self.classes.item(i).data(Qt.UserRole) == name:
                self.classes.setCurrentRow(i); return

    def _class_selected(self, it, _prev=None):
        if it:
            self.st.current_class = it.data(Qt.UserRole); self.st.classChanged.emit(self.st.current_class)

    def _edit_class(self):
        name, ok = QInputDialog.getText(self, "Class", "Class name:")
        if not ok or not name:
            return
        kind, ok = QInputDialog.getItem(self, "Class", "Kind:", ["defect", "normal", "appearance", "meta"], 0, False)
        if not ok:
            return
        color, _ = QInputDialog.getText(self, "Class", "Color (#rrggbb):", text="#ff00ff")
        sc, _ = QInputDialog.getText(self, "Class", "Shortcut key (optional):")
        ch, _ = QInputDialog.getItem(self, "Class", "Training channel:", ["", "black_spot", "bubble", "intensity"], 0, False)
        self.st.project.upsert_class(name, kind, color or "#ff00ff", sc, ch)
        self.load_classes()

    # ------------------------------------------------------------------ table
    def refresh(self):
        f = self.st.frame
        if not f:
            return
        anns = self.st.store.for_frame(f["id"])
        defects = {d["id"]: d["defect_uid"] for d in self.st.store.defects(f["bottle_id"])}
        self.table.blockSignals(True)
        self.table.setRowCount(len(anns))
        for r, a in enumerate(anns):
            vals = [a["id"], a["class_name"], a["label_state"], a["geometry_type"], a["visibility"],
                    "yes" if a["confirmed"] else "NO (suggested)", defects.get(a["defect_id"], ""), a["source"]]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.table.blockSignals(False)
        self.review.blockSignals(True); self.review.setCurrentText(f["review_status"]); self.review.blockSignals(False)
        self.scopes.blockSignals(True); self.scopes.clear()
        sc = set(self.st.store.review_scopes(f["id"]))
        for c in self.st.project.classes():
            if c["kind"] == "defect":
                it = QListWidgetItem(c["name"]); it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if c["name"] in sc else Qt.Unchecked); self.scopes.addItem(it)
        self.scopes.blockSignals(False)
        try:
            ov = self.st.store.overlaps(f["id"], (f["height"], f["width"]))
            self.overlap.setText("Overlaps: " + ", ".join(f"#{a}/#{b} {k}" for a, b, k in ov) if ov else "")
        except Exception:
            self.overlap.setText("")

    def selected_ids(self) -> list[int]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [int(self.table.item(r, 0).text()) for r in rows]

    def _sel(self):
        ids = self.selected_ids()
        self.st.selected_annotation = ids[0] if ids else None
        self.st.annotationsChanged.emit() if False else None

    # ------------------------------------------------------------------ actions
    def _each(self, fn):
        ids = self.selected_ids()
        if not ids:
            return
        for i in ids:
            fn(i)
        self.st.annotationsChanged.emit()

    def delete(self):
        self._each(self.st.store.delete)

    def confirm(self):
        self._each(self.st.store.confirm)

    def keyframe(self):
        def f(i):
            a = self.st.store.get(i)
            self.st.store.update(i, is_keyframe=1)
            if a["defect_id"]:
                self.st.store.update_defect(a["defect_id"], keyframe_id=a["frame_id"])
        self._each(f)

    def set_visibility(self):
        v = self.vis.currentText()
        self._each(lambda i: self.st.store.update(i, visibility=v))

    def set_verdict(self):
        v = self.verdict.currentText()
        self._each(lambda i: self.st.store.update(i, verdict=v))

    def new_defect(self):
        ids = self.selected_ids()
        if not ids:
            return
        a = self.st.store.get(ids[0])
        did = self.st.store.create_defect(self.st.bottle_id, a["class_name"], keyframe_id=a["frame_id"])
        for i in ids:
            self.st.store.update(i, defect_id=did)
        self.st.annotationsChanged.emit()

    def link_defect(self):
        ids = self.selected_ids()
        ds = self.st.store.defects(self.st.bottle_id)
        if not ids or not ds:
            return
        item, ok = QInputDialog.getItem(self, "Link", "Defect:", [f"{d['defect_uid']} {d['class_name']}" for d in ds], 0, False)
        if ok:
            did = ds[[f"{d['defect_uid']} {d['class_name']}" for d in ds].index(item)]["id"]
            for i in ids:
                self.st.store.update(i, defect_id=did)
            self.st.annotationsChanged.emit()

    def propagate(self):
        ids = self.selected_ids()
        if not ids:
            return
        n, ok = QInputDialog.getInt(self, "Copy to next frames", "N frames (negative = previous):", 2, -36, 36)
        if not ok or n == 0:
            return
        k = len(self.st.frames)
        step = 1 if n > 0 else -1
        targets = [self.st.frames[(self.st.index + step * j) % k]["id"] for j in range(1, abs(n) + 1)]
        for i in ids:
            self.st.store.propagate(i, targets, "copy")
        self.st.log.emit(f"copied {len(ids)} label(s) to {len(targets)} frames as UNCONFIRMED suggestions")

    def interpolate(self):
        """Select one label here; pick an end label of the same defect in another frame."""
        ids = self.selected_ids()
        if len(ids) != 1:
            QMessageBox.information(self, "Interpolate", "Select exactly one start label (with a Defect ID)."); return
        a = self.st.store.get(ids[0])
        if not a["defect_id"]:
            QMessageBox.information(self, "Interpolate", "The label needs a Defect ID."); return
        obs = [o for o in self.st.store.defect_observations(a["defect_id"]) if o["id"] != a["id"] and o["confirmed"]]
        if not obs:
            QMessageBox.information(self, "Interpolate", "No other confirmed observation of this defect."); return
        names = [f"#{o['id']} angle {o['angle_deg']}" for o in obs]
        item, ok = QInputDialog.getItem(self, "Interpolate", "End keyframe:", names, 0, False)
        if not ok:
            return
        e = obs[names.index(item)]
        fidx = {f["id"]: i for i, f in enumerate(self.st.frames)}
        i0, i1 = fidx.get(a["frame_id"]), fidx.get(e["frame_id"])
        if i0 is None or i1 is None:
            return
        lo, hi = sorted((i0, i1))
        targets = [self.st.frames[i]["id"] for i in range(lo + 1, hi)]
        start, end = (a, e) if i0 < i1 else (e, a)
        self.st.store.propagate(start["id"], targets, "interpolate", end["id"])
        self.st.log.emit(f"interpolated {len(targets)} UNCONFIRMED suggestions")

    def track(self):
        from ..core.image_io import load_mono
        from ..core.tracking import track_point_template
        ids = self.selected_ids()
        if len(ids) != 1:
            return
        a = self.st.store.get(ids[0])
        if a["geometry_type"] != "point":
            QMessageBox.information(self, "Track", "Template tracking supports point labels."); return
        f0 = self.st.frame; nxt = self.st.frames[(self.st.index + 1) % len(self.st.frames)]
        x, y, s = track_point_template(load_mono(f0["path"]), load_mono(nxt["path"]), a["geometry"]["x"], a["geometry"]["y"])
        self.st.store.add(nxt["id"], a["class_name"], a["label_state"], "point", {"x": x, "y": y, "r": a["geometry"].get("r", 2)},
                          defect_id=a["defect_id"], visibility="uncertain" if s < 0.6 else "visible", source="tracked",
                          confirmed=False, attributes={"ncc": s})
        self.st.log.emit(f"tracked to next frame (NCC={s:.2f}) as UNCONFIRMED suggestion")

    def _review_status(self, s):
        if self.st.frame:
            self.st.project.set_frame_review_status(self.st.frame["id"], s)

    def _scope(self, it):
        self.st.store.set_review_scope(self.st.frame["id"], it.text(), it.checkState() == Qt.Checked)


class TimelinePanel(QWidget):
    """Rotation sequence navigation + per-angle score graph."""

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.prev = QPushButton("◀ (←)"); self.next = QPushButton("(→) ▶")
        self.play = QPushButton("Play"); self.play.setCheckable(True)
        self.fps = QDoubleSpinBox(); self.fps.setRange(0.5, 60); self.fps.setValue(5); self.fps.setSuffix(" fps")
        self.angle = QDoubleSpinBox(); self.angle.setRange(-720, 720); self.angle.setDecimals(2); self.angle.setSuffix("°")
        self.angle.setKeyboardTracking(False)
        self.slider = QSlider(Qt.Horizontal)
        self.info = QLabel("-")
        for w in (self.prev, self.slider, self.next, QLabel("angle"), self.angle, self.play, self.fps, self.info):
            row.addWidget(w)
        row.setStretch(1, 1)
        lay.addLayout(row)
        self.plot = pg.PlotWidget(); self.plot.setBackground("#202020"); self.plot.setMaximumHeight(150)
        self.plot.setLabel("bottom", "angle (deg)"); self.plot.setXRange(0, 360)
        self.cur_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("y", width=1)); self.plot.addItem(self.cur_line)
        self.plot.scene().sigMouseClicked.connect(self._plot_click)
        lay.addWidget(self.plot)
        self.timer = QTimer(self); self.timer.timeout.connect(lambda: self.st.step(1))
        self.prev.clicked.connect(lambda: self.st.step(-1)); self.next.clicked.connect(lambda: self.st.step(1))
        self.slider.valueChanged.connect(lambda v: v != self.st.index and self.st.goto_index(v))
        self.angle.valueChanged.connect(lambda a: self.st.goto_angle(a))
        self.play.toggled.connect(self._play); self.fps.valueChanged.connect(lambda: self._play(self.play.isChecked()))
        st.sessionChanged.connect(self._session); st.frameChanged.connect(self._frame)
        self.series = {}

    def _play(self, on):
        self.timer.stop()
        if on:
            self.timer.start(int(1000 / self.fps.value()))

    def _session(self):
        self.slider.blockSignals(True); self.slider.setRange(0, max(0, len(self.st.frames) - 1)); self.slider.blockSignals(False)
        self.set_series({})

    def _frame(self):
        f = self.st.frame
        self.slider.blockSignals(True); self.slider.setValue(self.st.index); self.slider.blockSignals(False)
        a = f["angle_deg"] if f["angle_deg"] is not None else float(self.st.index)
        self.angle.blockSignals(True); self.angle.setValue(a); self.angle.blockSignals(False)
        self.cur_line.setValue(a)
        self.info.setText(f"frame {self.st.index + 1}/{len(self.st.frames)}  idx={f['frame_index']}")

    def set_series(self, series: dict[str, tuple[list, list]]):
        """name -> (angles, values). E.g. per-frame max candidate score, or a defect's angle profile."""
        self.plot.clear(); self.plot.addItem(self.cur_line)
        self.plot.addLegend()
        colors = ["c", "m", "g", "r", "w"]
        for i, (k, (x, y)) in enumerate(series.items()):
            self.plot.plot(x, y, pen=pg.mkPen(colors[i % len(colors)], width=1.5), symbol="o", symbolSize=4, name=k)

    def _plot_click(self, ev):
        p = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
        self.st.goto_angle(p.x())
