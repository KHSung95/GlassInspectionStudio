"""Print console stream health: fps, frame age, worker restarts, GenTL stream statistics, NIC throughput.

    python tools/stream_health.py [--port 8765] [--secs 20]
"""
import argparse
import http.client
import json
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--secs", type=int, default=20)
a = ap.parse_args()
conn = None
for _ in range(60):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=10)
        conn.request("GET", "/api/status"); conn.getresponse().read(); break
    except Exception:
        time.sleep(1)


def get(path):
    conn.request("GET", path)
    return json.loads(conn.getresponse().read())


first = get("/api/stream_stats")
t0 = time.time()
while time.time() - t0 < a.secs:
    s = get("/api/status")
    print(f"t+{int(time.time() - t0):3d}s fps {s['fps']:5} frames {s['frames']:6} age {s.get('frame_age_ms')} ms "
          f"restarts {s.get('reconnects')}", flush=True)
    time.sleep(2)
last = get("/api/stream_stats")
print("priority:", last.get("priority"), "| restarts:", last.get("worker_restarts"), last.get("restart_log"))
fs, ls = first.get("stream", {}), last.get("stream", {})
for k in sorted(ls):
    v0, v1 = fs.get(k), ls.get(k)
    d = (v1 - v0) if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) else ""
    print(f"  {k:40s} {v1!s:>12}  (+{d} in window)" if d != "" else f"  {k:40s} {v1!s:>12}")
