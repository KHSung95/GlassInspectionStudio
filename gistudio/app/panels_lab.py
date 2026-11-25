"""Classical Vision Lab / Zero-shot Inference Lab / supervised inference — one panel class per family."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..algorithms.base import create, registry
from ..algorithms.zero_shot.plugins import DEFAULT_PROMPTS
from ..core.image_io import load_mono
from ..core.project import Project
from ..core.splits import split_bottles, splits
from ..inference.engine import run_on_frames
from .state import AppState
from .widgets import ParamForm, Worker, error_box


class AlgorithmLab(QWidget):
    overlayRequested = Signal(object)      # np.ndarray score map or None
    previewCandidates = Signal(object)     # list[dict] (unsaved preview)

    def __init__(self, st: AppState, family: str, parent=None):
        super().__init__(parent)
        self.st, self.family = st, family
        self.worker = None
        self.last_maps: dict[str, np.ndarray] = {}
        lay = QVBoxLayout(self)
        self.alg = QComboBox()
        self.algs = {k: v for k, v in registry(family).items()}
        for k, v in self.algs.items():
            self.alg.addItem(v.display_name, k)
        self.alg.currentIndexChanged.connect(self._alg_changed)
        lay.addWidget(self.alg)
        self.sem = QLabel(); self.sem.setWordWrap(True); self.sem.setStyleSheet("color:#aaa"); lay.addWidget(self.sem)
        self.form_area = QScrollArea(); self.form_area.setWidgetResizable(True); lay.addWidget(self.form_area, 2)
        if family == "zero_shot":
            lay.addWidget(QLabel("Prompt set (JSON: {normal:[...], defect:[...]}) — used by OpenCLIP patch mode"))
            self.prompts = QPlainTextEdit(json.dumps(DEFAULT_PROMPTS, indent=1)); self.prompts.setMaximumHeight(140)
            lay.addWidget(self.prompts)
            pr = QHBoxLayout(); b1 = QPushButton("Save prompts…"); b1.clicked.connect(self._save_prompts)
            b2 = QPushButton("Load prompts…"); b2.clicked.connect(self._load_prompts); pr.addWidget(b1); pr.addWidget(b2)
            lay.addLayout(pr)
        row = QHBoxLayout()
        self.live = QCheckBox("Live preview"); self.live.setVisible(family == "classical")
        b = QPushButton("Run on frame"); b.clicked.connect(lambda: self.run_frame(save=True))
        broi = QPushButton("Run on ROI"); broi.setToolTip("Uses the last Measure-tool rectangle"); broi.clicked.connect(self.run_roi)
        for w in (self.live, b, broi):
            row.addWidget(w)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.scope = QComboBox(); self.scope.addItems(["current session", "current bottle", "all frames"])
        bb = QPushButton("Run batch"); bb.clicked.connect(self.run_batch)
        self.stop = QPushButton("Stop"); self.stop.clicked.connect(lambda: self.worker and self.worker.stop())
        for w in (self.scope, bb, self.stop):
            row2.addWidget(w)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        sp = QPushButton("Save preset…"); sp.clicked.connect(self._save_preset)
        lp = QPushButton("Load preset…"); lp.clicked.connect(self._load_preset)
        row3.addWidget(sp); row3.addWidget(lp)
        if family == "classical":
            sw = QPushButton("Sweep && recommend…")
            sw.setToolTip("Search parameters against confirmed labels and recommend presets")
            sw.clicked.connect(self.open_sweep)
            row3.addWidget(sw)
        lay.addLayout(row3)
        self.prog = QProgressBar(); lay.addWidget(self.prog)
        ov = QHBoxLayout(); self.map_cb = QComboBox(); self.map_cb.currentTextChanged.connect(self._show_map)
        self.opacity = QSlider(Qt.Horizontal); self.opacity.setRange(0, 100); self.opacity.setValue(60)
        self.opacity.valueChanged.connect(lambda: self._show_map(self.map_cb.currentText()))
        ov.addWidget(QLabel("Overlay")); ov.addWidget(self.map_cb, 1); ov.addWidget(self.opacity, 1); lay.addLayout(ov)
        self.info = QLabel(); self.info.setWordWrap(True); self.info.setStyleSheet("color:#ffcc66"); lay.addWidget(self.info)
        self.table = QTableWidget(0, 5); self.table.setHorizontalHeaderLabels(["score", "class", "x", "y", "area"])
        self.table.cellDoubleClicked.connect(self._goto_cand); lay.addWidget(self.table, 2)
        self._preview = []
        self.roi = None
        self._debounce = QTimer(self); self._debounce.setSingleShot(True); self._debounce.timeout.connect(
            lambda: self.run_frame(save=False))
        self._alg_changed()
        st.frameChanged.connect(lambda: self.live.isChecked() and self._debounce.start(150))

    # ------------------------------------------------------------------ algorithm/params
    def alg_id(self) -> str:
        return self.alg.currentData()

    def _alg_changed(self):
        cls = self.algs[self.alg_id()]
        self.sem.setText(f"Score: {cls.score_semantics}")
        self.form = ParamForm(cls.params_spec)
        self.form.changed.connect(lambda: self.live.isChecked() and self._debounce.start(250))
        self.form_area.setWidget(self.form)

    def params(self) -> dict:
        p = self.form.values()
        if self.family == "zero_shot" and self.alg_id() == "zeroshot.clip_patch":
            f = Path(self.st.project.root) / "prompts" / "_current.json"
            f.parent.mkdir(exist_ok=True)
            f.write_text(self.prompts.toPlainText(), encoding="utf-8")
            json.loads(self.prompts.toPlainText())  # validate
            p["prompts_json"] = str(f)
        return p

    # ------------------------------------------------------------------ presets / prompts
    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Preset", "Preset name:")
        if ok and name:
            d = Path(self.st.project.root) / "presets"; d.mkdir(exist_ok=True)
            (d / f"{name}.json").write_text(json.dumps({"algorithm": self.alg_id(), "params": self.form.values()},
                                                       indent=2), encoding="utf-8")
            self.st.log.emit(f"preset saved: {name}")

    def _load_preset(self):
        f, _ = QFileDialog.getOpenFileName(self, "Preset", str(Path(self.st.project.root) / "presets"), "*.json")
        if f:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            i = self.alg.findData(d["algorithm"])
            if i >= 0:
                self.alg.setCurrentIndex(i); self.form.set_values(d["params"])

    def _save_prompts(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save prompts", str(Path(self.st.project.root) / "prompts"), "*.json")
        if f:
            Path(f).write_text(self.prompts.toPlainText(), encoding="utf-8")

    def _load_prompts(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load prompts", str(Path(self.st.project.root) / "prompts"), "*.json")
        if f:
            self.prompts.setPlainText(Path(f).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ running
    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def run_frame(self, save: bool = True, roi: dict | None = None):
        f = self.st.frame
        if f is None or self._busy():
            return
        try:
            params = self.params()
        except Exception as e:
            error_box(self, "Parameters", repr(e)); return
        alg_id, root, fid = self.alg_id(), self.st.project.root, f["id"]

        def job(progress, stop):
            if save and roi is None:
                prj = Project(root)
                r = run_on_frames(prj, alg_id, [fid], params, True, progress, stop, notes="GUI single frame")
                from ..inference.engine import candidates, load_maps
                return {"run": r, "maps": load_maps(prj, r["id"], fid), "cands": candidates(prj, r["id"], fid)}
            img = load_mono(f["path"])
            ox = oy = 0
            meta = {"bit_depth": f.get("bit_depth"), "progress": progress}
            if roi is not None:
                from ..algorithms.classical.ops import noise_sigma
                meta["noise_sigma"] = noise_sigma(img)   # frame-level sigma: ROI scores match full-frame scores
                x0, y0 = int(max(0, roi["x0"])), int(max(0, roi["y0"]))
                x1, y1 = int(min(img.shape[1], roi["x1"])), int(min(img.shape[0], roi["y1"]))
                img, ox, oy = img[y0:y1, x0:x1], x0, y0
            res = create(alg_id, **params).run(img, meta)
            cands = []
            for c in res.candidates:
                g = dict(c.geometry)
                for k in ("x", "x0", "x1"):
                    if k in g:
                        g[k] += ox
                for k in ("y", "y0", "y1"):
                    if k in g:
                        g[k] += oy
                cands.append({"id": -1, "score": c.score, "class_name": c.class_name, "geometry_type": c.geometry_type,
                              "geometry": g, "area_px": c.area_px, "source_algorithm": c.source_algorithm,
                              "review_status": "preview"})
            maps = {}
            for k, m in res.score_maps.items():
                full = np.zeros((f["height"], f["width"]), np.float32)
                full[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
                maps[k] = full
            return {"run": None, "maps": maps, "cands": cands, "ms": res.inference_ms, "warnings": res.warnings}

        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda i, n, m: (self.prog.setMaximum(n), self.prog.setValue(i)))
        self.worker.done.connect(self._done)
        self.worker.failed.connect(lambda tb: error_box(self, "Algorithm failed", tb))
        self.worker.start()

    def run_roi(self):
        if self.roi is None:
            self.info.setText("Draw a rectangle with the Measure tool (M) first."); return
        self.run_frame(save=False, roi=self.roi)

    def _done(self, res):
        self.last_maps = res["maps"]
        cur = self.map_cb.currentText()
        self.map_cb.blockSignals(True); self.map_cb.clear(); self.map_cb.addItems(["(none)"] + list(self.last_maps))
        self.map_cb.setCurrentText(cur if cur in self.last_maps else (list(self.last_maps)[0] if self.last_maps else "(none)"))
        self.map_cb.blockSignals(False)
        self._show_map(self.map_cb.currentText())
        cands = res["cands"]
        self._preview = cands
        self.table.setRowCount(len(cands))
        for r, c in enumerate(cands[:2000]):
            from ..core.geometry import geometry_center
            x, y = geometry_center(c["geometry_type"], c["geometry"])
            for col, v in enumerate([f"{c['score']:.4g}", c["class_name"], f"{x:.1f}", f"{y:.1f}", f"{c.get('area_px', 0):.0f}"]):
                it = QTableWidgetItem(v); it.setData(Qt.UserRole, (x, y)); self.table.setItem(r, col, it)
        if res["run"] is not None:
            s = res["run"].get("summary", {})
            self.st.current_run_id = res["run"]["id"]
            self.info.setText(f"Run #{res['run']['id']} saved: {len(cands)} candidates, {s.get('ms_mean', 0):.1f} ms\n"
                              + "\n".join(s.get("warnings", [])))
            self.st.runsChanged.emit(); self.st.candidatesChanged.emit()
        else:
            self.info.setText(f"Preview (not saved): {len(cands)} candidates, {res.get('ms', 0):.1f} ms\n"
                              + "\n".join(res.get("warnings", [])))
            self.previewCandidates.emit(cands)

    def _show_map(self, name):
        m = self.last_maps.get(name)
        self.overlayRequested.emit(None if m is None else (m, self.opacity.value() / 100.0))

    def _goto_cand(self, r, _c):
        x, y = self.table.item(r, 0).data(Qt.UserRole)
        self.parent_viewer_center(x, y)

    def parent_viewer_center(self, x, y):  # replaced by main window
        pass

    def open_sweep(self):
        if self.st.project is None:
            return
        dlg = SweepDialog(self.st, self.alg_id(), self)
        dlg.applyParams.connect(self._apply_recommended)
        dlg.show()

    def _apply_recommended(self, alg_id: str, params: dict):
        i = self.alg.findData(alg_id)
        if i >= 0:
            self.alg.setCurrentIndex(i)
            self.form.set_values(params)
            self.st.log.emit(f"applied recommended parameters for {alg_id}: {params}")

    def frames_for_scope(self) -> list[int]:
        s = self.scope.currentText()
        if s == "current session":
            return [f["id"] for f in self.st.frames]
        if s == "current bottle":
            return [f["id"] for f in self.st.project.all_frames([self.st.bottle_id])]
        return [f["id"] for f in self.st.project.all_frames()]

    def run_batch(self):
        if self._busy() or self.st.project is None:
            return
        try:
            params = self.params()
        except Exception as e:
            error_box(self, "Parameters", repr(e)); return
        fids = self.frames_for_scope()
        alg_id, root = self.alg_id(), self.st.project.root

        def job(progress, stop):
            return run_on_frames(Project(root), alg_id, fids, params, True, progress, stop, notes="GUI batch")
        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda i, n, m: (self.prog.setMaximum(n), self.prog.setValue(i), self.st.log.emit(m)))
        self.worker.done.connect(lambda r: (setattr(self.st, "current_run_id", r["id"]), self.st.runsChanged.emit(),
                                            self.st.candidatesChanged.emit(),
                                            self.info.setText(f"Run #{r['id']} {r['status']}: {r.get('summary', {})}")))
        self.worker.failed.connect(lambda tb: error_box(self, "Batch failed", tb))
        self.worker.start()


class SweepDialog(QWidget):
    """Parameter sweep against confirmed labels -> recommended presets (balanced / high recall / low FP / zero FP)."""
    applyParams = Signal(str, dict)

    def __init__(self, st: AppState, alg_id: str, parent=None):
        super().__init__(parent, Qt.Window)
        from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QSpinBox
        from ..evaluation.param_sweep import SPACES
        self.st = st
        self.setWindowTitle("Parameter sweep & preset recommendation")
        self.resize(1100, 700)
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.alg = QComboBox()
        for k, v in registry("classical").items():
            self.alg.addItem(v.display_name, k)
        self.alg.setCurrentIndex(max(0, self.alg.findData(alg_id)))
        self.scope = QComboBox(); self.scope.addItems(["labelled frames of current bottle", "all labelled frames"])
        self.space = QPlainTextEdit(); self.space.setMaximumHeight(90)
        self.alg.currentIndexChanged.connect(lambda: self.space.setPlainText(json.dumps(SPACES.get(self.alg.currentData(), {}))))
        self.space.setPlainText(json.dumps(SPACES.get(self.alg.currentData(), {})))
        self.tol = QDoubleSpinBox(); self.tol.setValue(6.0); self.tol.setSuffix(" px")
        self.margin = QSpinBox(); self.margin.setRange(0, 4000); self.margin.setValue(160)
        self.margin.setToolTip("Crop margin around labels (0 = full frames; exhaustively reviewed frames always full)")
        self.maxc = QSpinBox(); self.maxc.setRange(1, 2000); self.maxc.setValue(60)
        self.rt = QDoubleSpinBox(); self.rt.setRange(0, 1); self.rt.setSingleStep(0.05); self.rt.setValue(0.9)
        self.fpb = QDoubleSpinBox(); self.fpb.setRange(0, 100); self.fpb.setDecimals(3); self.fpb.setValue(0.2)
        self.hold = QDoubleSpinBox(); self.hold.setRange(0, 0.8); self.hold.setSingleStep(0.1); self.hold.setValue(0.0)
        for n, w in (("Algorithm", self.alg), ("Frames", self.scope), ("Search space (JSON, threshold is swept automatically)", self.space),
                     ("Point tolerance", self.tol), ("Crop margin", self.margin), ("Max combinations", self.maxc),
                     ("Recall target (high-recall preset)", self.rt), ("FP budget / frame (low-FP preset)", self.fpb),
                     ("Bottle hold-out fraction", self.hold)):
            f.addRow(n, w)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.brun = QPushButton("Run sweep"); self.brun.clicked.connect(self.run)
        self.bstop = QPushButton("Stop"); self.bstop.clicked.connect(lambda: self.worker and self.worker.stop())
        self.bapply = QPushButton("Apply selected to lab"); self.bapply.clicked.connect(self.apply)
        self.bsave = QPushButton("Save all as presets"); self.bsave.clicked.connect(self.save)
        for b in (self.brun, self.bstop, self.bapply, self.bsave):
            row.addWidget(b)
        lay.addLayout(row)
        self.prog = QProgressBar(); lay.addWidget(self.prog)
        self.warn = QLabel(); self.warn.setWordWrap(True); self.warn.setStyleSheet("color:#ffaa66"); lay.addWidget(self.warn)
        self.table = QTableWidget(0, 0); lay.addWidget(self.table, 1)
        self.worker = None
        self.result = None

    def run(self):
        from ..evaluation.param_sweep import SweepConfig, labelled_frames, run_sweep
        p = self.st.project
        if self.scope.currentIndex() == 0 and self.st.bottle_id is not None:
            fids = labelled_frames(p, [self.st.bottle_id])
        else:
            fids = labelled_frames(p)
        if not fids:
            error_box(self, "Sweep", "No frames with confirmed labels in this scope."); return
        try:
            space = json.loads(self.space.toPlainText() or "{}")
        except Exception as e:
            error_box(self, "Search space", repr(e)); return
        cfg = SweepConfig(self.alg.currentData(), fids, space or None, None, self.tol.value(), self.margin.value(),
                          self.maxc.value(), 0, self.rt.value(), self.fpb.value(), self.hold.value())
        root = p.root

        def job(progress, stop):
            return run_sweep(Project(root), cfg, progress, stop)
        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda i, n, m: (self.prog.setMaximum(n), self.prog.setValue(i)))
        self.worker.done.connect(self.show_result)
        self.worker.failed.connect(lambda tb: error_box(self, "Sweep failed", tb))
        self.worker.start()

    def show_result(self, r):
        self.result = r
        cols = ["preset", "threshold", "params", "recall", "precision(verified)", "F1", "FP verified/frame",
                "unverified/frame", "hold-out recall", "hold-out FP/frame"]
        rows = [("defaults (baseline)", r["baseline_defaults"]["params"], r["baseline_defaults"]["tune"], None)] + \
               [(x["objective"], x["params"], x["tune"], x.get("holdout")) for x in r["recommendations"]]
        self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols); self.table.setRowCount(len(rows))
        for i, (name, params, m, h) in enumerate(rows):
            short = {k: v for k, v in params.items() if k not in ("threshold", "max_candidates", "max_area")}
            vals = [name, params.get("threshold"), json.dumps(short), m["recall"], m["precision_verified"], m["f1_verified"],
                    m["fp_per_frame"], m["unverified_per_frame"], (h or {}).get("recall"), (h or {}).get("fp_per_frame")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
                it.setData(Qt.UserRole, params)
                self.table.setItem(i, c, it)
        self.table.resizeColumnsToContents()
        self.warn.setText(f"GT={r['n_gt']} on {r['tune_frames']} frames (hold-out {r['holdout_frames']}), "
                          f"{r['combos_evaluated']} combos, {r['seconds']} s.  " + " ".join(r["warnings"]) + "  " + r["note"])

    def apply(self):
        it = self.table.currentItem()
        if it is not None and self.result:
            self.applyParams.emit(self.result["algorithm"], it.data(Qt.UserRole))

    def save(self):
        from ..evaluation.param_sweep import save_presets
        if self.result:
            paths = save_presets(self.st.project, self.result)
            self.st.log.emit("saved presets:\n" + "\n".join(paths))
            self.warn.setText(self.warn.text() + f"\nSaved {len(paths)} presets to project/presets.")
