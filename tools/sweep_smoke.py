"""Headless check of the Sweep & recommend dialog.  python tools/sweep_smoke.py <project_copy> <screenshot.png>"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gistudio.app.panels_lab as pl  # noqa: E402
from gui_smoke import pump  # noqa: E402

pl.error_box = lambda parent, title, text: print("ERROR BOX:", title, text, flush=True)
from gistudio.app.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication([])
    w = MainWindow(sys.argv[1]); w.show(); pump(500)
    lab = w.classical
    dlg = pl.SweepDialog(w.st, "classical.blackhat", None)
    dlg.applyParams.connect(lab._apply_recommended)
    dlg.show()
    dlg.maxc.setValue(6); dlg.scope.setCurrentIndex(0)
    dlg.run()
    for _ in range(1800):
        pump(100)
        if dlg.worker is not None and not dlg.worker.isRunning():
            break
    pump(300)
    print("rows:", dlg.table.rowCount(), "|", dlg.warn.text()[:160], flush=True)
    dlg.table.setCurrentCell(1, 0); dlg.apply(); pump(100)
    print("lab params after apply:", lab.form.values(), flush=True)
    dlg.grab().save(sys.argv[2])
    dlg.close(); w.close()


if __name__ == "__main__":
    main()
