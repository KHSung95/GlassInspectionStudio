"""Measure GenTL stream buffering: buffer handling mode / count and how many stale frames are queued.

    python tools/latency_probe.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gistudio.acquisition.genicam_camera import GenICamCamera  # noqa: E402


def main():
    cam = GenICamCamera()
    cam.open()
    ia = cam._ia
    ds = ia.data_streams[0] if hasattr(ia, "data_streams") else None
    print("harvesters num_buffers:", getattr(ia, "num_buffers", None))
    if ds is not None:
        nm = ds.node_map
        for n in ("StreamBufferHandlingMode", "StreamBufferCountMode", "StreamBufferCountManual",
                  "StreamBufferCountResult", "StreamDefaultBufferCount"):
            try:
                print(n, "=", getattr(nm, n).value)
            except Exception as e:
                print(n, "n/a", type(e).__name__)
    cam.start()
    # steady-state fetch timing
    t = []
    for _ in range(10):
        t0 = time.time(); cam.grab(); t.append(time.time() - t0)
    print("steady fetch s:", [round(x, 3) for x in t])
    # idle 5 s, then see how many frames come back instantly (= queued stale frames)
    time.sleep(5)
    fast = 0
    for _ in range(40):
        t0 = time.time(); cam.grab(); dt = time.time() - t0
        if dt < 0.03:
            fast += 1
        else:
            break
    print("frames returned instantly after 5 s idle (stale queue depth):", fast)
    cam.close()


if __name__ == "__main__":
    main()
