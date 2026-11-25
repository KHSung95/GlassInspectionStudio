"""Headless GUI smoke test: opens a project, navigates, annotates, runs a classical preview, saves screenshots.

    python tools/gui_smoke.py <project_dir> <out_dir>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gistudio.app.main_window import MainWindow, dark_palette  # noqa: E402


def pump(ms=200):
    loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()


def wait_worker(lab, timeout_ms=120000):
    t = 0
    while lab.worker is not None and lab.worker.isRunning() and t < timeout_ms:
        pump(100); t += 100
    pump(300)


def main():
    proj, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    dark_palette(app)
    w = MainWindow(proj); w.resize(1700, 1000); w.show(); pump(800)
    assert w.st.frame is not None, "no frame loaded"
    w.st.step(1); w.st.step(-2); pump()
    assert w.st.index == len(w.st.frames) - 1, "circular navigation failed"
    w.st.goto_angle(365.0); pump()
    assert abs((w.st.frame["angle_deg"] or 0) - 0.0) < 1e-6 or abs(w.st.frame["angle_deg"] - 360) < 11
    n0 = len(w.st.store.for_frame(w.st.frame["id"]))
    w._shape("point", {"x": 100.5, "y": 120.5, "r": 2}, None)
    w._shape("normal_roi", {"x0": 10, "y0": 10, "x1": 40, "y1": 40}, None)
    assert len(w.st.store.for_frame(w.st.frame["id"])) == n0 + 2
    w.undo(); assert len(w.st.store.for_frame(w.st.frame["id"])) == n0 + 1
    w.redo(); assert len(w.st.store.for_frame(w.st.frame["id"])) == n0 + 2
    w.viewer.center_on_image(320, 260, 3.0); pump()
    w.grab().save(str(out / "gui_annotation.png"))
    w.docks["Classical Vision Lab"].raise_()
    w.classical.run_frame(save=False); wait_worker(w.classical)
    print("preview candidates:", len(w.preview_cands))
    w.viewer.fit(); pump()
    w.grab().save(str(out / "gui_classical.png"))
    w.compare_mode.setCurrentText("previous frame"); pump(500)
    w.grab().save(str(out / "gui_compare.png"))
    w.compare_mode.setCurrentText("off")
    w.viewer.center_on_image(320, 200, 60.0); pump()
    w.grab().save(str(out / "gui_pixels.png"))
    # training launched from the GUI runs in a separate process; GUI stays responsive
    from PySide6.QtCore import QProcess
    tr = w.training
    tr.refresh()
    if tr.split.count():
        tr.epochs.setValue(1); tr.spe.setValue(8); tr.batch.setValue(2); tr.tile.setCurrentText("128"); tr.base.setValue(8)
        tr._start(); pump(500)
        t = 0
        while tr.proc.state() != QProcess.NotRunning and t < 300000:
            pump(200); t += 200
        print("gui training exit:", tr.proc.exitCode(), "best.pt exists:", (tr.run_dir / "best.pt").exists())
    # evaluation panel on the first run
    ev = w.evaluation; ev.refresh()
    if ev.runs.count():
        ev.runs.item(ev.runs.count() - 1).setSelected(True)
        ev.evaluate()
        t = 0
        while ev.worker.isRunning() and t < 300000:
            pump(200); t += 200
        pump(300)
        print("eval rows:", ev.table.rowCount(), "warnings:", ev.warn.text()[:120])
    w.docks["Evaluation Lab"].raise_(); pump()
    w.grab().save(str(out / "gui_eval.png"))
    print("OK", w.st.project.stats())
    w.close()


if __name__ == "__main__":
    main()
