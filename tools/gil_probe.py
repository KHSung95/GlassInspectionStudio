"""Does the camera grab loop starve other Python threads (GIL held inside the GenTL fetch)?"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.acquisition.genicam_camera import GenICamCamera  # noqa: E402


def spin(seconds):
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        n += 1
    return n


base = spin(2.0)
cam = GenICamCamera(); cam.open(); cam.start()
stop = False
grabs = []


def loop():
    while not stop:
        t = time.time(); cam.grab(); grabs.append(time.time() - t)


th = threading.Thread(target=loop, daemon=True); th.start()
time.sleep(1)
busy = spin(2.0)
stop = True; th.join(10); cam.close()
print(f"main-thread loop iterations: idle {base}  while grabbing {busy}  ratio {busy / base:.2f}")
print(f"grabs {len(grabs)}  mean fetch {sum(grabs) / max(1, len(grabs)):.3f}s")
