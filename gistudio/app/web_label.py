"""Browser-based draft labelling for a project (no camera needed).

    python -m gistudio.cli label <project> [--port 8770]

* 127.0.0.1 only. Full-resolution frames are served as display JPEGs; geometry is saved in ORIGINAL image
  coordinates through the same AnnotationStore as the desktop GUI (history, undo-safe soft delete).
* Operator marks are human labels (confirmed) tagged attributes.draft = true so a second review pass can be
  distinguished. Labels proposed by the AI/pilot (attributes.provisional_by) are shown dashed and can be
  confirmed (attributes.reviewed_by = operator) or deleted.
"""
from __future__ import annotations

import json
import re
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from ..core.annotations import AnnotationStore
from ..core.image_io import load_mono
from ..core.preprocessing import DisplayTransform
from ..core.project import Project

PAGE = Path(__file__).with_name("web_label.html")
EXTRA_CLASSES = [("Marker Ink", "normal", "#38d9ff", "M", "")]   # customer pen marks: negative for training


class LabelServer:
    def __init__(self, project: Project, author: str = "operator"):
        self.p = project
        self.author = author
        for name, kind, color, sc, ch in EXTRA_CLASSES:
            if name not in project.class_map():
                project.upsert_class(name, kind, color, sc, ch)
        self._img_cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def store(self) -> AnnotationStore:
        return AnnotationStore(self.p, author=self.author)

    def frames(self) -> list[dict]:
        out = []
        for f in sorted(self.p.all_frames(), key=lambda f: (f["bottle_code"], f["session_code"], f["path"])):
            n = self.p.db.one("SELECT COUNT(*), SUM(label_state='positive') FROM annotations WHERE frame_id=? AND deleted=0",
                              (f["id"],))
            out.append({"id": f["id"], "name": Path(f["path"]).name, "bottle": f["bottle_code"],
                        "session": f["session_code"], "angle": f["angle_deg"], "w": f["width"], "h": f["height"],
                        "review_status": f["review_status"], "n_ann": n[0], "n_pos": n[1] or 0})
        return out

    def image(self, fid: int, mode: str) -> bytes:
        key = (fid, mode)
        with self._lock:
            if key in self._img_cache:
                self._img_cache.move_to_end(key)
                return self._img_cache[key]
        f = self.p.frame(fid)
        img = load_mono(f["path"])
        dt = DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=f.get("bit_depth")) if mode == "stretch" \
            else DisplayTransform(bit_depth=f.get("bit_depth"))
        ok, enc = cv2.imencode(".jpg", dt.apply(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
        b = enc.tobytes()
        with self._lock:
            self._img_cache[key] = b
            while len(self._img_cache) > 6:
                self._img_cache.popitem(last=False)
        return b

    def anns(self, fid: int) -> list[dict]:
        return self.store().for_frame(fid)

    def add(self, fid: int, d: dict) -> dict:
        st = self.store()
        attrs = {"draft": True}
        if d.get("note"):
            attrs["note"] = d["note"]
        aid = st.add(fid, d["class_name"], d.get("label_state", "positive"), d["geometry_type"], d["geometry"],
                     visibility=d.get("visibility", "visible"), source="manual", attributes=attrs)
        return st.get(aid)

    def update(self, aid: int, d: dict) -> dict:
        st = self.store()
        a = st.get(aid)
        if a is None:
            raise KeyError(aid)
        changes = {k: d[k] for k in ("class_name", "label_state", "visibility", "verdict", "geometry") if k in d}
        if d.get("review") == "confirm":
            changes["attributes"] = {**a["attributes"], "reviewed_by": self.author}
            changes["confirmed"] = 1
        st.update(aid, **changes)
        return st.get(aid)

    def delete(self, aid: int) -> None:
        self.store().delete(aid)


def make_handler(srv: LabelServer):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # keep-alive (new loopback connections are unreliable on this workstation)

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/frames":
                    self._json(srv.frames())
                elif u.path == "/api/classes":
                    self._json(srv.p.classes())
                elif m := re.fullmatch(r"/api/frame/(\d+)/image\.jpg", u.path):
                    self._send(200, srv.image(int(m[1]), q.get("mode", "stretch")), "image/jpeg")
                elif m := re.fullmatch(r"/api/frame/(\d+)/anns", u.path):
                    self._json(srv.anns(int(m[1])))
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
            except Exception as e:
                self._json({"ok": False, "error": repr(e), "trace": traceback.format_exc()}, 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0) or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
            try:
                if m := re.fullmatch(r"/api/frame/(\d+)/anns", u.path):
                    self._json({"ok": True, "ann": srv.add(int(m[1]), d)})
                elif m := re.fullmatch(r"/api/frame/(\d+)/review", u.path):
                    srv.p.set_frame_review_status(int(m[1]), d["status"]); self._json({"ok": True})
                elif m := re.fullmatch(r"/api/ann/(\d+)/delete", u.path):
                    srv.delete(int(m[1])); self._json({"ok": True})
                elif m := re.fullmatch(r"/api/ann/(\d+)", u.path):
                    self._json({"ok": True, "ann": srv.update(int(m[1]), d)})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._json({"ok": False, "error": repr(e)}, 400)
    return H


def serve(project_dir: str, port: int = 8770, author: str = "operator") -> None:
    p = Project(project_dir)
    srv = LabelServer(p, author)
    http = ThreadingHTTPServer(("127.0.0.1", port), make_handler(srv))
    http.daemon_threads = True
    print(f"Labeling on http://127.0.0.1:{port}  (project {p.root})", flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
