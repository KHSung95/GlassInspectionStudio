"""Training Lab, Evaluation Lab (+ angular analysis) and Active Learning Review panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from ..core.project import Project
from ..core.splits import make_split, split_bottles, split_id_by_name, splits
from .state import AppState
from .widgets import Worker, error_box

REPO = Path(__file__).resolve().parents[2]


def _frames_spec(project: Project, spec: str) -> list[int]:
    if spec == "all frames":
        return [f["id"] for f in project.all_frames()]
    name, subset = spec.rsplit(":", 1)
    return [f["id"] for f in project.all_frames(split_bottles(project, split_id_by_name(project, name), subset))]


def _subset_choices(project: Project) -> list[str]:
    out = ["all frames"]
    for s in splits(project):
        out += [f"{s['name']}:{k}" for k in ("train", "val", "test")]
    return out


class TrainingPanel(QWidget):
    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        self.proc: QProcess | None = None
        self.run_dir: Path | None = None
        lay = QVBoxLayout(self)
        sg = QGroupBox("Bottle-level split (physical sample ID; no leakage)"); sl = QFormLayout(sg)
        self.sp_name = QLineEdit("split-1"); self.sp_ratio = QLineEdit("0.6,0.2,0.2"); self.sp_seed = QSpinBox()
        self.sp_hold = QLineEdit(); self.sp_hold.setPlaceholderText("product types held out to test (comma)")
        b = QPushButton("Create split"); b.clicked.connect(self._make_split)
        for n, w in (("name", self.sp_name), ("ratios", self.sp_ratio), ("seed", self.sp_seed), ("hold-out", self.sp_hold)):
            sl.addRow(n, w)
        sl.addRow(b)
        self.split_info = QLabel(); self.split_info.setWordWrap(True); sl.addRow(self.split_info)
        lay.addWidget(sg)
        tg = QGroupBox("U-Net training (separate process)"); f = QFormLayout(tg)
        self.split = QComboBox(); self.model = QComboBox(); self.model.addItems(["U-Net (HRUNet)", "Multi-scale Patch CNN"])
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10000); self.epochs.setValue(30)
        self.batch = QSpinBox(); self.batch.setRange(1, 512); self.batch.setValue(8)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-6, 1); self.lr.setValue(1e-3)
        self.tile = QComboBox(); self.tile.addItems(["256", "512", "1024", "128"])
        self.spe = QSpinBox(); self.spe.setRange(8, 10**6); self.spe.setValue(512)
        self.base = QSpinBox(); self.base.setRange(4, 128); self.base.setValue(16)
        self.loss = QLineEdit(json.dumps({"black_spot": "heatmap_focal", "bubble": "bce+dice", "intensity": "bce+dice"}))
        self.sigma = QDoubleSpinBox(); self.sigma.setRange(0.3, 10); self.sigma.setValue(1.5)
        self.device = QComboBox(); self.device.addItems(["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
        self.amp = QCheckBox("Mixed precision (CUDA only)"); self.seed = QSpinBox(); self.seed.setRange(0, 10**6)
        self.resume = QLineEdit(); self.resume.setPlaceholderText("optional last.pt to resume")
        for n, w in (("split", self.split), ("model", self.model), ("epochs", self.epochs), ("batch", self.batch),
                     ("learning rate", self.lr), ("tile", self.tile), ("samples/epoch", self.spe),
                     ("base channels", self.base), ("loss per channel", self.loss),
                     ("point heatmap sigma (px)", self.sigma), ("device", self.device), ("", self.amp),
                     ("seed", self.seed), ("resume", self.resume)):
            f.addRow(n, w)
        row = QHBoxLayout(); self.start = QPushButton("Start training"); self.start.clicked.connect(self._start)
        self.stop = QPushButton("Stop (graceful)"); self.stop.clicked.connect(self._stop)
        row.addWidget(self.start); row.addWidget(self.stop); f.addRow(row)
        lay.addWidget(tg)
        self.prog = QProgressBar(); lay.addWidget(self.prog)
        self.plot = pg.PlotWidget(); self.plot.setBackground("#202020"); self.plot.addLegend(); self.plot.setLabel("bottom", "epoch")
        lay.addWidget(self.plot, 1)
        self.status = QLabel(); self.status.setWordWrap(True); lay.addWidget(self.status)
        self.hist = {"train": [], "val": []}
        st.projectChanged.connect(self.refresh); st.runsChanged.connect(self.refresh)

    def refresh(self):
        p = self.st.project
        self.split.clear(); self.split.addItems([s["name"] for s in splits(p)])
        self.split_info.setText("\n".join(f"{s['name']}: {s['counts']}" for s in splits(p)))

    def _make_split(self):
        try:
            hold = [h.strip() for h in self.sp_hold.text().split(",") if h.strip()]
            make_split(self.st.project, self.sp_name.text(), tuple(float(x) for x in self.sp_ratio.text().split(",")),
                       self.sp_seed.value(), holdout_product_types=hold)
            self.refresh(); self.st.runsChanged.emit()
        except Exception as e:
            error_box(self, "Split", repr(e))

    def _start(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            return
        p = self.st.project
        if not self.split.currentText():
            QMessageBox.information(self, "Training", "Create a split first."); return
        unet = self.model.currentIndex() == 0
        run = p.create_run("train", "supervised.unet" if unet else "supervised.patch_cnn",
                           params={"split": self.split.currentText()})
        self.run_dir = run["output_dir"]; self.run_id = run["id"]
        if unet:
            from ..core.label_targets import TargetPolicy
            cfg = {"epochs": self.epochs.value(), "batch_size": self.batch.value(), "lr": self.lr.value(),
                   "tile": int(self.tile.currentText()), "samples_per_epoch": self.spe.value(),
                   "base_channels": self.base.value(), "loss": json.loads(self.loss.text()), "device": self.device.currentText(),
                   "amp": self.amp.isChecked(), "seed": self.seed.value(),
                   "policy": {**TargetPolicy().to_dict(), "point_sigma": self.sigma.value()}}
            args = ["-m", "gistudio.cli", "train-unet", str(p.root), "--split", self.split.currentText(),
                    "--run-dir", str(self.run_dir), "--config", json.dumps(cfg)]
            if self.resume.text():
                args += ["--resume", self.resume.text()]
        else:
            cfg = {"epochs": self.epochs.value(), "batch_size": self.batch.value(), "lr": self.lr.value(),
                   "device": self.device.currentText(), "seed": self.seed.value()}
            args = ["-m", "gistudio.cli", "train-patch", str(p.root), "--split", self.split.currentText(),
                    "--run-dir", str(self.run_dir), "--config", json.dumps(cfg)]
        self.hist = {"train": [], "val": []}; self.plot.clear()
        self.proc = QProcess(self); self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.start(sys.executable, args)
        self.st.log.emit(f"training started: {self.run_dir}")

    def _read(self):
        for line in bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").splitlines():
            if line.startswith("@@PROGRESS "):
                r = json.loads(line[11:])
                if r.get("event") == "step":
                    self.prog.setMaximum(r["steps"]); self.prog.setValue(r["step"] + 1)
                elif r.get("event") == "epoch":
                    self.hist["train"].append(r["train_loss"])
                    if r.get("val_loss") is not None:
                        self.hist["val"].append(r["val_loss"])
                    self.plot.clear(); self.plot.plot(self.hist["train"], pen="c", name="train")
                    if self.hist["val"]:
                        self.plot.plot(self.hist["val"], pen="m", name="val")
                    self.status.setText(json.dumps({k: v for k, v in r.items() if k != "t"}, default=str))
                elif r.get("event") in ("start", "done", "stopped", "resumed"):
                    self.st.log.emit(f"train: {r}")
            elif line.strip():
                self.st.log.emit(line)

    def _stop(self):
        if self.run_dir:
            (Path(self.run_dir) / "STOP").write_text("stop", encoding="utf-8")
            self.st.log.emit("stop requested (finishes current step, checkpoint kept)")

    def _finished(self, code, _status):
        self.st.project.finish_run(self.run_id, "done" if code == 0 else "failed")
        self.st.log.emit(f"training process exited with code {code}; checkpoint: {Path(self.run_dir) / 'best.pt'}")
        self.st.runsChanged.emit()


class EvaluationPanel(QWidget):
    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        self.worker = None
        lay = QVBoxLayout(self)
        tabs = QTabWidget(); lay.addWidget(tabs)
        # ---- evaluation tab
        ev = QWidget(); el = QVBoxLayout(ev)
        self.runs = QListWidget(); self.runs.setSelectionMode(QAbstractItemView.ExtendedSelection); self.runs.setMaximumHeight(150)
        el.addWidget(QLabel("Runs to compare (same test frames)")); el.addWidget(self.runs)
        f = QFormLayout(); self.subset = QComboBox(); self.tol = QDoubleSpinBox(); self.tol.setValue(6.0)
        self.thr = QLineEdit(); self.thr.setPlaceholderText("operating thresholds per run (comma) or empty")
        f.addRow("Frames", self.subset); f.addRow("Point tolerance (px)", self.tol); f.addRow("Thresholds", self.thr)
        el.addLayout(f)
        row = QHBoxLayout(); b = QPushButton("Evaluate"); b.clicked.connect(self.evaluate)
        rep = QPushButton("Export report…"); rep.clicked.connect(self.export); row.addWidget(b); row.addWidget(rep)
        el.addLayout(row)
        self.warn = QLabel(); self.warn.setWordWrap(True); self.warn.setStyleSheet("color:#ff9966"); el.addWidget(self.warn)
        self.table = QTableWidget(0, 0); el.addWidget(self.table, 1)
        self.froc = pg.PlotWidget(); self.froc.setBackground("#202020"); self.froc.addLegend()
        self.froc.setLabel("bottom", "verified FP / frame"); self.froc.setLabel("left", "recall"); el.addWidget(self.froc, 1)
        tabs.addTab(ev, "Model comparison")
        # ---- angular tab
        an = QWidget(); al = QVBoxLayout(an); af = QFormLayout()
        self.a_run = QComboBox(); self.a_thr = QDoubleSpinBox(); self.a_thr.setDecimals(4); self.a_thr.setRange(0, 1e6); self.a_thr.setValue(0.5)
        self.a_steps = QLineEdit("1,2,5,10,15,30,90"); self.a_rev = QDoubleSpinBox(); self.a_rev.setRange(0.01, 1000); self.a_rev.setValue(1.0)
        af.addRow("Run (score source)", self.a_run); af.addRow("Detection threshold", self.a_thr)
        af.addRow("Re-sampling steps (deg)", self.a_steps); af.addRow("Time per revolution (s)", self.a_rev)
        al.addLayout(af)
        ab = QPushButton("Analyse current bottle"); ab.clicked.connect(self.angular); al.addWidget(ab)
        self.a_table = QTableWidget(0, 0); al.addWidget(self.a_table, 1)
        self.a_defects = QListWidget(); self.a_defects.itemClicked.connect(self._defect_plot); al.addWidget(self.a_defects)
        self.s_table = QTableWidget(0, 0); al.addWidget(self.s_table, 1)
        tabs.addTab(an, "Angular visibility")
        # ---- multiview tab
        mv = QWidget(); ml = QVBoxLayout(mv); mf = QFormLayout()
        self.m_run = QComboBox(); self.m_thr = QDoubleSpinBox(); self.m_thr.setDecimals(4); self.m_thr.setRange(0, 1e6); self.m_thr.setValue(0.5)
        self.m_contrast = QDoubleSpinBox(); self.m_contrast.setRange(-1e6, 1e6); self.m_contrast.setSpecialValueText("off")
        self.m_contrast.setMinimum(-1); self.m_contrast.setValue(-1)
        mf.addRow("Run", self.m_run); mf.addRow("Score threshold", self.m_thr); mf.addRow("Min local contrast", self.m_contrast)
        ml.addLayout(mf)
        mb = QPushButton("Aggregate current bottle (+ save NG evidence)"); mb.clicked.connect(self.aggregate); ml.addWidget(mb)
        self.m_out = QPlainTextEdit(); self.m_out.setReadOnly(True); ml.addWidget(self.m_out)
        tabs.addTab(mv, "Multi-view")
        self.results = []
        self.profiles = []
        st.projectChanged.connect(self.refresh); st.runsChanged.connect(self.refresh)

    def refresh(self):
        p = self.st.project
        self.runs.clear()
        for combo in (self.a_run, self.m_run):
            combo.clear()
        self.a_run.addItem("(annotation visibility only)", None)
        for r in p.runs():
            if r["kind"] in ("classical", "zeroshot", "infer"):
                it = QListWidgetItem(f"#{r['id']} {r['algorithm']} [{r['status']}] {r['created_at']} {r['notes']}")
                it.setData(Qt.UserRole, r["id"]); self.runs.addItem(it)
                self.a_run.addItem(f"#{r['id']} {r['algorithm']}", r["id"]); self.m_run.addItem(f"#{r['id']} {r['algorithm']}", r["id"])
        cur = self.subset.currentText(); self.subset.clear(); self.subset.addItems(_subset_choices(p))
        if cur:
            self.subset.setCurrentText(cur)

    def evaluate(self):
        from ..evaluation.metrics import EvalConfig, evaluate_run
        ids = [i.data(Qt.UserRole) for i in self.runs.selectedItems()]
        if not ids:
            return
        thrs = [float(x) for x in self.thr.text().split(",") if x.strip()]
        root, subset, tol = self.st.project.root, self.subset.currentText(), self.tol.value()

        def job(progress, stop):
            prj = Project(root); fr = _frames_spec(prj, subset); out = []
            for k, rid in enumerate(ids):
                out.append(evaluate_run(prj, rid, fr, EvalConfig(tolerance_px=tol,
                                                                 operating_threshold=thrs[k] if k < len(thrs) else None)))
                progress(k + 1, len(ids), f"run {rid}")
            return out
        self.worker = Worker(job, self)
        self.worker.done.connect(self._show)
        self.worker.failed.connect(lambda tb: error_box(self, "Evaluation failed", tb))
        self.worker.start()

    def _show(self, res):
        self.results = res
        cols = ["run", "algorithm", "thr", "recall", "precision(verified)", "FP verified", "unverified", "FP/frame",
                "loc err", "phys. defect recall", "frame recall", "bottle recall", "false reject", "ms/frame", "labels"]
        self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols); self.table.setRowCount(len(res))
        fmt = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
        self.froc.clear()
        for r, e in enumerate(res):
            o = e["operating_point"]
            vals = [e["run_id"], e["algorithm"], o["threshold"], o["recall"], o["precision_verified"], o["fp_verified"],
                    o["unverified"], o["fp_per_frame"], o["loc_error_mean"], e["physical_defects"]["recall_any_view"],
                    e["frame_level"]["recall"], e["bottle_level"]["recall"],
                    e["bottle_level"]["false_reject_rate"] if e["bottle_level"]["false_reject_rate"] is not None else "N/A",
                    e["timing"]["frame_ms_mean"], e["label_completeness"]["status"]]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(fmt(v)))
            pts = [(c["fp_per_frame"], c["recall"]) for c in e["froc"] if c["recall"] is not None]
            if pts:
                pts.sort()
                self.froc.plot([p[0] for p in pts], [p[1] for p in pts], pen=pg.intColor(r), symbol="o", symbolSize=4,
                               name=f"#{e['run_id']} {e['algorithm']}")
        self.warn.setText("\n".join(sorted({w for e in res for w in e["warnings"]})))

    def export(self):
        from ..evaluation.metrics import write_report
        if not self.results:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Report", str(self.st.project.root / "exports" / "evaluation.md"), "*.md")
        if f:
            write_report(self.results, f)
            Path(f).with_suffix(".json").write_text(json.dumps(self.results, indent=1, default=str), encoding="utf-8")
            self.st.log.emit(f"report written: {f}")

    # ------------------------------------------------------------------ angular
    def angular(self):
        from ..evaluation.angular import defect_profile, required_capture, sampling_experiment, visible_intervals
        if self.st.bottle_id is None:
            return
        run = self.a_run.currentData(); thr = self.a_thr.value()
        steps = [float(s) for s in self.a_steps.text().split(",") if s.strip()]
        root, bid = self.st.project.root, self.st.bottle_id

        def job(progress, stop):
            prj = Project(root)
            from ..core.annotations import AnnotationStore
            ds = AnnotationStore(prj).defects(bid)
            profs = []
            for i, d in enumerate(ds):
                profs.append(defect_profile(prj, d["id"], run)); progress(i + 1, len(ds), d["defect_uid"])
            key = "score" if run else "visibility"
            return profs, sampling_experiment(profs, steps, key=key, thr=thr), \
                [visible_intervals(p, key, thr) for p in profs]
        w = Worker(job, self)

        def done(res):
            profs, samp, ints = res
            self.profiles = profs
            cols = ["defect", "class", "annotated/frames", "best view (score)", "best view (contrast)", "visible arcs (deg)"]
            self.a_table.setColumnCount(len(cols)); self.a_table.setHorizontalHeaderLabels(cols); self.a_table.setRowCount(len(profs))
            self.a_defects.clear()
            for r, (p, iv) in enumerate(zip(profs, ints)):
                vals = [p["defect_uid"], p["class_name"], f"{p['coverage']['annotated']}/{p['coverage']['frames']}",
                        json.dumps(p["best_view_by_score"]), json.dumps(p["best_view_by_contrast"]),
                        "; ".join(f"{a:.0f}–{b:.0f}" for a, b in iv)]
                for c, v in enumerate(vals):
                    self.a_table.setItem(r, c, QTableWidgetItem(str(v)))
                self.a_defects.addItem(p["defect_uid"])
            cols = ["step (deg)", "frames/rev", "required fps", "mean miss rate", "worst-start miss rate"]
            self.s_table.setColumnCount(len(cols)); self.s_table.setHorizontalHeaderLabels(cols); self.s_table.setRowCount(len(samp))
            for r, (k, v) in enumerate(samp.items()):
                rc = required_capture(float(k), self.a_rev.value())
                for c, x in enumerate([k, v["frames_per_rev"], f"{rc['fps']:.1f}", v["mean_miss_rate"], v["worst_miss_rate"]]):
                    self.s_table.setItem(r, c, QTableWidgetItem("-" if x is None else str(x)))
        w.done.connect(done); w.failed.connect(lambda tb: error_box(self, "Angular analysis failed", tb))
        self._aw = w; w.start()

    def _defect_plot(self, it):
        p = next((x for x in self.profiles if x["defect_uid"] == it.text()), None)
        if p is None or not hasattr(self, "timeline"):
            return
        rows = [r for r in p["rows"] if r["annotated"] and r["angle"] is not None]
        series = {"local contrast": ([r["angle"] for r in rows], [r["local_contrast"] for r in rows]),
                  "size px": ([r["angle"] for r in rows], [r["size_px"] for r in rows])}
        if rows and "score" in rows[0]:
            series["score"] = ([r["angle"] for r in rows], [r["score"] for r in rows])
        self.timeline.set_series(series)

    # ------------------------------------------------------------------ multiview
    def aggregate(self):
        from ..inference.multiview import AggregationConfig, aggregate_bottle, save_evidence
        run = self.m_run.currentData()
        if run is None or self.st.bottle_id is None:
            return
        cfg = AggregationConfig(score_threshold=self.m_thr.value(),
                                min_local_contrast=None if self.m_contrast.value() < 0 else self.m_contrast.value())
        agg = aggregate_bottle(self.st.project, run, self.st.bottle_id, cfg)
        paths = save_evidence(self.st.project, agg, self.st.project.root / "exports" / f"evidence_run{run}")
        self.m_out.setPlainText(json.dumps(agg, indent=1, default=str) + "\n\nEvidence:\n" + "\n".join(paths))
        self.st.candidatesChanged.emit()


class ReviewPanel(QWidget):
    """Active-learning review queue. Accept -> positive GT (ai_accepted); Reject -> hard negative; Unknown."""

    def __init__(self, st: AppState, parent=None):
        super().__init__(parent)
        self.st = st
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.run = QComboBox(); self.other = QComboBox()
        self.strategy = QComboBox()
        from ..inference.candidate_management import STRATEGIES
        self.strategy.addItems(STRATEGIES)
        self.thr = QDoubleSpinBox(); self.thr.setDecimals(4); self.thr.setRange(-1e6, 1e6); self.thr.setValue(0.5)
        self.group = QCheckBox("Apply decision to the whole multi-view group"); self.group.setChecked(True)
        self.cls = QComboBox()
        for n, w in (("Run", self.run), ("Strategy", self.strategy), ("Threshold", self.thr), ("Other run", self.other),
                     ("Accept as class", self.cls)):
            f.addRow(n, w)
        f.addRow(self.group)
        lay.addLayout(f)
        b = QPushButton("Load queue"); b.clicked.connect(self.load); lay.addWidget(b)
        self.list = QListWidget(); self.list.currentItemChanged.connect(self._goto); lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for t, d in (("Accept (A)", "accept"), ("Reject → hard negative (X)", "reject"), ("Unknown (U)", "unknown")):
            bt = QPushButton(t); bt.clicked.connect(lambda _=False, d=d: self.decide(d)); row.addWidget(bt)
        lay.addLayout(row)
        self.note = QLabel("AI candidates are NOT ground truth until accepted here."); self.note.setWordWrap(True)
        lay.addWidget(self.note)
        st.projectChanged.connect(self.refresh); st.runsChanged.connect(self.refresh)

    def refresh(self):
        p = self.st.project
        self.run.clear(); self.other.clear(); self.other.addItem("(none)", None)
        for r in p.runs():
            if r["kind"] in ("classical", "zeroshot", "infer"):
                self.run.addItem(f"#{r['id']} {r['algorithm']}", r["id"]); self.other.addItem(f"#{r['id']} {r['algorithm']}", r["id"])
        self.cls.clear(); self.cls.addItem("(candidate class)", None)
        for c in p.classes():
            if c["kind"] == "defect":
                self.cls.addItem(c["name"], c["name"])

    def load(self):
        from ..inference.candidate_management import review_queue
        if self.run.currentData() is None:
            return
        try:
            q = review_queue(self.st.project, self.run.currentData(), self.strategy.currentText(), self.thr.value(),
                             self.other.currentData())
        except Exception as e:
            error_box(self, "Queue", repr(e)); return
        self.list.clear()
        for c in q:
            it = QListWidgetItem(f"#{c['id']} f{c['frame_id']} {c['class_name']} score={c['score']:.4g} "
                                 f"{c.get('reason', '')} group={c['group_key'] or '-'}")
            it.setData(Qt.UserRole, c); self.list.addItem(it)
        self.st.log.emit(f"review queue: {len(q)} candidates")

    def _goto(self, it, _prev=None):
        if not it:
            return
        c = it.data(Qt.UserRole)
        self.st.selected_candidate = c["id"]
        self.st.goto_frame_id(c["frame_id"])
        from ..core.geometry import geometry_center
        x, y = geometry_center(c["geometry_type"], c["geometry"])
        if hasattr(self, "center"):
            self.center(x, y)

    def decide(self, d: str):
        it = self.list.currentItem()
        if not it:
            return
        c = it.data(Qt.UserRole)
        from ..inference.candidate_management import apply_group_review
        if self.group.isChecked():
            apply_group_review(self.st.project, c["id"], d, self.cls.currentData(), self.st.author)
        else:
            if d == "accept":
                self.st.store.accept_candidate(c["id"], self.cls.currentData())
            elif d == "reject":
                self.st.store.reject_candidate(c["id"])
            else:
                self.st.store.mark_candidate_unknown(c["id"])
        row = self.list.row(it); self.list.takeItem(row)
        self.st.annotationsChanged.emit(); self.st.candidatesChanged.emit()
        if self.list.count():
            self.list.setCurrentRow(min(row, self.list.count() - 1))
