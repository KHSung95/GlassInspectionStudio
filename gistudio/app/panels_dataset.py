"""Dataset Manager panels: bottle tree with filters, import dialog, metadata editor, thumbnails."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QSpinBox, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.dataset_import import ImportOptions, compare_sessions, find_duplicates, import_folder
from ..core.project import Project
from ..core.thumbnails import thumbnail
from .state import AppState
from .widgets import Worker, error_box


class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import image folder")
        lay = QFormLayout(self)
        self.folder = QLineEdit(); b = QPushButton("…"); b.clicked.connect(self._pick)
        row = QHBoxLayout(); row.addWidget(self.folder); row.addWidget(b)
        lay.addRow("Folder", row)
        self.layout_cb = QComboBox(); self.layout_cb.addItems(["bottle/session", "bottle", "flat"])
        lay.addRow("Layout", self.layout_cb)
        self.dataset = QLineEdit("default"); lay.addRow("Dataset", self.dataset)
        self.bottle = QLineEdit(); self.bottle.setPlaceholderText("(flat layout only)"); lay.addRow("Bottle code", self.bottle)
        self.step = QDoubleSpinBox(); self.step.setRange(0, 360); self.step.setDecimals(3); self.step.setSpecialValueText("from filename/CSV")
        lay.addRow("Angle step (fallback)", self.step)
        self.csv = QLineEdit(); self.csv.setPlaceholderText("optional: CSV filename,angle[,frame_index]")
        lay.addRow("Angle CSV", self.csv)
        self.regex = QLineEdit(ImportOptions().angle_regex); lay.addRow("Angle regex", self.regex)
        self.bits = QSpinBox(); self.bits.setRange(0, 16); self.bits.setSpecialValueText("unknown")
        lay.addRow("Sensor bit depth", self.bits)
        self.camera = QLineEdit(); lay.addRow("Camera model", self.camera)
        self.lighting = QLineEdit(); lay.addRow("Lighting", self.lighting)
        self.mmpx = QDoubleSpinBox(); self.mmpx.setDecimals(5); self.mmpx.setRange(0, 10); self.mmpx.setSpecialValueText("unknown")
        lay.addRow("mm / px", self.mmpx)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); lay.addRow(bb)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Image folder")
        if d:
            self.folder.setText(d)

    def options(self) -> ImportOptions:
        sm = {}
        if self.bits.value():
            sm["bit_depth"] = self.bits.value()
        if self.camera.text():
            sm["camera_model"] = self.camera.text()
        if self.lighting.text():
            sm["lighting"] = self.lighting.text()
        if self.mmpx.value():
            sm["mm_per_px"] = self.mmpx.value()
        return ImportOptions(layout=self.layout_cb.currentText(), dataset_name=self.dataset.text() or "default",
                             bottle_code=self.bottle.text(), angle_step=self.step.value() or None,
                             angle_csv=self.csv.text() or None, angle_regex=self.regex.text(), session_meta=sm)


class DatasetPanel(QWidget):
    """Bottle → Session tree, search/filter, thumbnails of the current session."""

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        f = QHBoxLayout()
        self.search = QLineEdit(); self.search.setPlaceholderText("search bottle / lot / memo")
        self.ptype = QComboBox(); self.verdict = QComboBox()
        self.verdict.addItems(["", "unknown", "ng", "ok", "under_review"])
        for w in (self.search, self.ptype, self.verdict):
            f.addWidget(w)
        lay.addLayout(f)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Bottle / Session", "Type", "Frames", "Status"])
        self.tree.itemDoubleClicked.connect(self._open)
        self.tree.itemClicked.connect(self._open)
        lay.addWidget(self.tree, 3)
        self.thumbs = QListWidget(); self.thumbs.setViewMode(QListWidget.IconMode); self.thumbs.setIconSize(QPixmap(96, 96).size())
        self.thumbs.setResizeMode(QListWidget.Adjust); self.thumbs.setMovement(QListWidget.Static)
        self.thumbs.setGridSize(QPixmap(110, 124).size())
        self.thumbs.itemClicked.connect(lambda it: self.st.goto_index(it.data(Qt.UserRole)))
        lay.addWidget(self.thumbs, 2)
        self._thumb_queue: list = []
        self._timer = QTimer(self); self._timer.timeout.connect(self._load_thumb_chunk)
        self.search.textChanged.connect(self.refresh); self.ptype.currentTextChanged.connect(self.refresh)
        self.verdict.currentTextChanged.connect(self.refresh)
        st.projectChanged.connect(self._project); st.sessionChanged.connect(self._session_thumbs)
        st.frameChanged.connect(self._mark_current)

    def _project(self):
        self.ptype.blockSignals(True); self.ptype.clear(); self.ptype.addItems([""] + self.st.project.product_types())
        self.ptype.blockSignals(False)
        self.refresh()

    def refresh(self):
        p = self.st.project
        if p is None:
            return
        self.tree.clear()
        for ds in p.datasets():
            di = QTreeWidgetItem([ds["name"], "", "", ""]); di.setFlags(di.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(di)
            for b in p.bottles(ds["id"], self.search.text(), self.ptype.currentText(), self.verdict.currentText()):
                bi = QTreeWidgetItem([b["bottle_code"], b["product_type"], "", b["verdict"]])
                bi.setData(0, Qt.UserRole, ("bottle", b["id"]))
                di.addChild(bi)
                for s in p.sessions(b["id"]):
                    n = p.db.one("SELECT COUNT(*), SUM(review_status='reviewed') FROM frames WHERE session_id=?", (s["id"],))
                    si = QTreeWidgetItem([s["session_code"], s["camera_model"] or "", str(n[0]),
                                          f"reviewed {n[1] or 0}/{n[0]}"])
                    si.setData(0, Qt.UserRole, ("session", s["id"]))
                    bi.addChild(si)
            di.setExpanded(True)

    def _open(self, item, _col=0):
        d = item.data(0, Qt.UserRole)
        if not d:
            return
        if d[0] == "session" and d[1] != self.st.session_id:
            self.st.set_session(d[1])
        elif d[0] == "bottle":
            ss = self.st.project.sessions(d[1])
            if ss and ss[0]["id"] != self.st.session_id:
                self.st.set_session(ss[0]["id"])

    def _session_thumbs(self):
        self.thumbs.clear(); self._thumb_queue = []
        for i, f in enumerate(self.st.frames):
            it = QListWidgetItem(f"{f['angle_deg']:.1f}°" if f["angle_deg"] is not None else f"#{f['frame_index']}")
            it.setData(Qt.UserRole, i); it.setSizeHint(QPixmap(104, 118).size())
            blank = QPixmap(96, 96); blank.fill(Qt.black); it.setIcon(QIcon(blank))
            self.thumbs.addItem(it)
            self._thumb_queue.append((it, f))
        self._timer.start(1)

    def _load_thumb_chunk(self):
        """Lazy thumbnail loading (a few per tick) from the on-disk cache."""
        p = self.st.project
        bd = p.session(self.st.session_id).get("bit_depth") if self.st.session_id else None
        for _ in range(3):
            if not self._thumb_queue:
                self._timer.stop(); return
            it, f = self._thumb_queue.pop(0)
            try:
                t = thumbnail(p.thumbs_dir(), f["sha256"], f["path"], 96, bd)
                qi = QImage(t.data, t.shape[1], t.shape[0], t.strides[0], QImage.Format_Grayscale8).copy()
                it.setIcon(QIcon(QPixmap.fromImage(qi)))
            except Exception as e:
                it.setText(f"ERR {e}")

    def _mark_current(self):
        if 0 <= self.st.index < self.thumbs.count():
            self.thumbs.setCurrentRow(self.st.index)


class PropertiesPanel(QWidget):
    """Editable bottle + session metadata and frame info."""

    BOTTLE = ["product_type", "variety", "lot_no", "physical_sample_id", "verdict", "memo"]
    SESSION = ["captured_at", "camera_model", "bit_depth", "exposure_us", "gain_db", "lighting", "lens",
               "magnification", "mm_per_px", "memo"]

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        tabs = QTabWidget(); lay.addWidget(tabs)
        bw = QWidget(); self.bf = QFormLayout(bw); self.be = {}
        for k in self.BOTTLE:
            if k == "verdict":
                e = QComboBox(); e.addItems(["unknown", "ng", "ok", "under_review"])
            else:
                e = QLineEdit()
            self.be[k] = e; self.bf.addRow(k, e)
        sb = QPushButton("Save bottle"); sb.clicked.connect(self._save_bottle); self.bf.addRow(sb)
        tabs.addTab(bw, "Bottle")
        sw = QWidget(); self.sf = QFormLayout(sw); self.se = {}
        for k in self.SESSION:
            e = QLineEdit(); self.se[k] = e; self.sf.addRow(k, e)
        ss = QPushButton("Save session"); ss.clicked.connect(self._save_session); self.sf.addRow(ss)
        tabs.addTab(sw, "Session")
        self.frame_info = QLabel(); self.frame_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.frame_info.setWordWrap(True)
        tabs.addTab(self.frame_info, "Frame")
        st.sessionChanged.connect(self._load); st.frameChanged.connect(self._frame)

    def _load(self):
        p = self.st.project
        b = p.bottle(self.st.bottle_id); s = p.session(self.st.session_id)
        for k, e in self.be.items():
            (e.setCurrentText if isinstance(e, QComboBox) else e.setText)(str(b.get(k) or ""))
        for k, e in self.se.items():
            e.setText("" if s.get(k) is None else str(s.get(k)))

    def _save_bottle(self):
        vals = {k: (e.currentText() if isinstance(e, QComboBox) else e.text()) for k, e in self.be.items()}
        self.st.project.update_bottle(self.st.bottle_id, **vals)
        self.st.log.emit("bottle metadata saved")

    def _save_session(self):
        vals = {}
        for k, e in self.se.items():
            t = e.text().strip()
            if k in ("bit_depth",):
                vals[k] = int(t) if t else None
            elif k in ("exposure_us", "gain_db", "magnification", "mm_per_px"):
                vals[k] = float(t) if t else None
            else:
                vals[k] = t
        self.st.project.update_session(self.st.session_id, **vals)
        self.st.log.emit("session metadata saved")

    def _frame(self):
        f = self.st.frame
        if f:
            keys = ["id", "bottle_code", "session_code", "frame_index", "angle_deg", "width", "height", "dtype",
                    "container_bits", "bit_depth", "mm_per_px", "review_status", "sha256", "path"]
            self.frame_info.setText("\n".join(f"{k}: {f.get(k)}" for k in keys))


def run_import(parent, st: AppState, on_done) -> Worker | None:
    dlg = ImportDialog(parent)
    if dlg.exec() != QDialog.Accepted or not dlg.folder.text():
        return None
    opts = dlg.options(); folder = dlg.folder.text(); root = st.project.root

    def job(progress, stop):
        prj = Project(root)  # own connection in this thread
        return import_folder(prj, folder, opts, progress)
    w = Worker(job, parent)
    w.progress.connect(lambda i, n, m: st.log.emit(f"import {i}/{n} {m}") if i == n or i % 20 == 0 else None)
    w.done.connect(lambda rep: (QMessageBox.information(parent, "Import", rep.summary() + (
        "\n\nDuplicates (skipped):\n" + "\n".join(f"{a}  ==  {b}" for a, b in rep.duplicates[:10]) if rep.duplicates else "") + (
        "\n\nErrors:\n" + "\n".join(f"{a}: {b}" for a, b in rep.errors[:10]) if rep.errors else "")), on_done()))
    w.failed.connect(lambda tb: error_box(parent, "Import failed", tb))
    w.start()
    return w


def show_duplicates(parent, st: AppState):
    d = find_duplicates(st.project)
    QMessageBox.information(parent, "Duplicate frames (same SHA-256)",
                            "\n\n".join("\n".join(g) for g in d[:30]) or "No duplicates found.")


def show_session_compare(parent, st: AppState):
    if st.bottle_id is None:
        return
    ss = st.project.sessions(st.bottle_id)
    if len(ss) < 2:
        QMessageBox.information(parent, "Compare sessions", "This bottle has only one capture session."); return
    r = compare_sessions(st.project, ss[0]["id"], ss[1]["id"])
    QMessageBox.information(parent, f"{ss[0]['session_code']} vs {ss[1]['session_code']}",
                            json.dumps(r, indent=1, default=str))
