"""Measure the web console's live-view latency (frame age at send time + fetch time).

    python tools/live_latency.py [--port 8765] [--n 15]
"""
import argparse
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--n", type=int, default=15)
a = ap.parse_args()
last = -1
t_start = time.time()
for _ in range(a.n):
    t = time.time()
    r = urllib.request.urlopen(f"http://127.0.0.1:{a.port}/live.jpg?after={last}", timeout=10)
    body = r.read()
    last = int(r.headers["X-Frame"])
    age = int(r.headers["X-Frame-Age-Ms"])
    fetch = int((time.time() - t) * 1000)
    print(f"frame {last:5d}  age at send {age:4d} ms  fetch {fetch:4d} ms  {len(body) // 1024} KB")
print(f"display rate {a.n / (time.time() - t_start):.1f} fps")
