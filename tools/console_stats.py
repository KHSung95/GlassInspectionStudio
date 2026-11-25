"""Poll the web console status for N seconds and print timing fields."""
import http.client
import json
import sys
import time
import urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
secs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for _ in range(60):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3).read(); break
    except Exception:
        time.sleep(0.7)
t0 = time.time()
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)   # keep-alive, like a browser
last = -1
while time.time() - t0 < secs:
    t = time.time()
    conn.request("GET", f"/live.jpg?after={last}")
    r = conn.getresponse(); r.read(); last = int(r.headers["X-Frame"])
    live_ms = int((time.time() - t) * 1000)
    print(f"live.jpg frame {last} age-at-send {r.headers['X-Frame-Age-Ms']} ms fetch {live_ms} ms", flush=True)
    t = time.time()
    conn.request("GET", "/api/status")
    s = json.loads(conn.getresponse().read())
    print(f"status rtt {int((time.time() - t) * 1000):5d} ms | fps {s['fps']:5} frames {s['frames']:5} "
          f"age {s.get('frame_age_ms')} ms grab {s.get('last_grab_ms')} ms jpeg {s.get('last_jpeg_ms')} ms "
          f"mode {s.get('stream_mode')} err {s['error'][:60]}", flush=True)
    time.sleep(1)
