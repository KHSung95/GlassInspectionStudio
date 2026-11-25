"""Generate CPU load (classical CV on real frames) for N seconds to test console robustness."""
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.algorithms.base import create  # noqa: E402
from gistudio.core.image_io import read_image  # noqa: E402

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20
files = sorted(glob.glob(r"C:\dev\data\GLASS_INSPECTION-2D\GLASS_INSPECTION\*.png"))[:3]
imgs = [read_image(f) for f in files]
alg = create("classical.log_blob", threshold=8.0)
t0, n = time.time(), 0
while time.time() - t0 < secs:
    alg.run(imgs[n % len(imgs)]); n += 1
print("load frames processed:", n)
