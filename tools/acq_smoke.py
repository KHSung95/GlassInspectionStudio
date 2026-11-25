"""Headless check of the Acquisition panel with the mock camera.  python tools/acq_smoke.py <project_copy>"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

import gistudio.app.panels_acquisition as pa  # noqa: E402
from gui_smoke import pump  # noqa: E402

pa.error_box = lambda parent, title, text: print("ERROR BOX:", title, text, flush=True)

from gistudio.app.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication([])
    w = MainWindow(sys.argv[1]); w.show(); pump(500)
    a = w.acq
    a.kind.setCurrentIndex(1); a.connect_camera(); print("connected", a.info.text(), flush=True)
    a.blive.setChecked(True); pump(1500)
    print("live qa:", a.qa.text(), flush=True)
    a.blive.setChecked(False); print("live stopped", flush=True)
    a.bottle.setText("MOCK-01"); a.step.setValue(30); a.burst.setValue(3)
    a.capture(); print("capture started", flush=True)
    for _ in range(300):
        pump(100)
        if a.worker is not None and not a.worker.isRunning():
            break
    pump(300)
    print("frames:", a.writer.count, "current:", w.st.frame["bottle_code"], w.st.frame["angle_deg"], flush=True)
    w.close()


if __name__ == "__main__":
    main()
