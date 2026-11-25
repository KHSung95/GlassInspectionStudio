"""Web console core (no HTTP server): live JPEG, native crop, session + averaged capture with the mock camera."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

from gistudio.acquisition.mock import MockCamera, MockRotationStage
from gistudio.acquisition.web_console import Console, make_handler
from gistudio.core.image_io import read_image


def test_console_capture_and_http(project):
    cam = MockCamera(MockRotationStage(), 200, 150); cam.open()
    con = Console(project, cam, live_width=100)
    try:
        con.wait_new(0, 10)
        assert con.live_jpeg()[:2] == b"\xff\xd8"
        png, meta = con.crop_jpeg(0.5, 0.5, 64, 2)
        assert png[:4] == b"\x89PNG" and meta["size"] == 64
        assert con.capture({})["ok"] is False                       # no session yet
        con.new_session({"bottle": "W-01", "start": 350})
        r = con.capture({"step": 20, "shots": 2, "average": 3})
        assert r["ok"] and r["count"] == 2 and r["next_angle"] == 390
        fr = project.frames(con.writer.session_id)
        assert [f["angle_deg"] for f in fr] == [350.0, 370.0]
        img = read_image(fr[0]["path"])
        assert img.dtype == np.uint16 and img.shape == (150, 200)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(con))
        th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
        port = srv.server_address[1]
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10).read())
        assert st["session"]["count"] == 2 and len(st["captures"]) == 2
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().startswith(b"<!doctype html>")
        srv.shutdown(); srv.server_close()
    finally:
        con.close()
