"""Local camera console: live MJPEG feed + native-resolution zoom + capture into a project, in any browser.

    python -m gistudio.cli console <project> [--port 8765] [--mock]

* Binds to 127.0.0.1 only (no remote access). No images leave this machine.
* One process owns the camera (close SpinView / the desktop GUI's camera connection first).
* Camera features are written ONLY when the operator presses an Apply button in the page.
* Live/zoom views are display-stretched JPEGs; captures are the raw frames (lossless TIFF) registered in the project.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..core.preprocessing import DisplayTransform
from ..core.project import Project
from .capture_session import CaptureSessionWriter, CaptureTarget

PAGE = Path(__file__).with_name("web_console.html")


def quality(img: np.ndarray, bits: int) -> dict:
    h, w = img.shape
    c = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4][::2, ::2].astype(np.float32)
    full = float((1 << bits) - 1)
    sub = img[::4, ::4]
    return {"focus": round(float(cv2.Laplacian(c / full, cv2.CV_32F).var() * 1e6), 2),
            "saturated_pct": round(float((sub >= full).mean() * 100), 3),
            "dark_pct": round(float((sub <= full * 0.02).mean() * 100), 2),
            "mean": round(float(sub.mean()), 1), "p50": float(np.percentile(sub, 50)),
            "p99": float(np.percentile(sub, 99)), "max": float(sub.max()), "full_scale": full}


class Console:
    def __init__(self, project: Project, camera, live_width: int = 1280):
        self.p = project
        self.cam = camera
        self.live_width = live_width
        self.lock = threading.Lock()          # camera access
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.frame_t = 0.0
        self.last_grab_ms = 0.0
        self.last_jpeg_ms = 0.0
        self.count = 0
        self.fps = 0.0
        self.jpeg: bytes | None = None
        self.jpeg_count = -1
        self.stretch = True
        self.paused = False
        self.running = True
        self.error = ""
        self.writer: CaptureSessionWriter | None = None
        self.next_angle = 0.0
        self.captures: list[dict] = []
        self.info = self._info()
        self.bits = self.info.get("bit_depth") or 8
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _info(self) -> dict:
        try:
            return self.cam.info()
        except Exception as e:
            return {"error": repr(e)}

    # ------------------------------------------------------------------ grab loop
    def _loop(self):
        last = time.time()
        while self.running:
            if self.paused:
                time.sleep(0.05); continue
            try:
                tg = time.perf_counter()
                with self.lock:
                    img = self.cam.grab(timeout_s=5.0)
                self.last_grab_ms = (time.perf_counter() - tg) * 1000
                self.error = ""
            except Exception as e:
                self.error = repr(e); time.sleep(0.5); continue
            with self.cond:
                self.frame = img
                self.frame_t = time.time()
                self.count += 1
                self.cond.notify_all()
            # never spin faster than ~30 fps (fast/mock sources would starve the HTTP threads)
            dt = time.time() - last
            if dt < 1 / 30:
                time.sleep(1 / 30 - dt)
            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(1e-3, now - last)); last = now

    def wait_new(self, after: int, timeout: float = 10.0) -> tuple[np.ndarray, int]:
        with self.cond:
            ok = self.cond.wait_for(lambda: self.count > after and self.frame is not None, timeout)
            if not ok:
                raise TimeoutError("no new frame from camera")
            return self.frame, self.count

    def _disp(self, img: np.ndarray) -> np.ndarray:
        if self.stretch:
            return DisplayTransform(stretch_percentiles=(0.5, 99.5), bit_depth=self.bits).apply(img)
        return DisplayTransform(bit_depth=self.bits).apply(img)

    def live_jpeg(self) -> bytes:
        with self.cond:
            img, n = self.frame, self.count
        if img is None:
            return b""
        if n != self.jpeg_count:
            tj = time.perf_counter()
            s = self.live_width / img.shape[1]
            small = cv2.resize(img, (self.live_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", self._disp(small), [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.jpeg, self.jpeg_count = enc.tobytes(), n
            self.last_jpeg_ms = (time.perf_counter() - tj) * 1000
        return self.jpeg

    def crop_jpeg(self, fx: float, fy: float, size: int, scale: int) -> tuple[bytes, dict]:
        """Native-resolution crop around a normalized position (fx, fy), upscaled with nearest neighbour."""
        with self.cond:
            img = self.frame
        if img is None:
            return b"", {}
        h, w = img.shape
        size = int(max(16, min(size, min(h, w))))
        cx, cy = int(fx * w), int(fy * h)
        x0, y0 = int(np.clip(cx - size // 2, 0, w - size)), int(np.clip(cy - size // 2, 0, h - size))
        c = img[y0:y0 + size, x0:x0 + size]
        lo, hi = np.percentile(c, (0.5, 99.5))
        d = DisplayTransform(window=max(1.0, hi - lo), level=(lo + hi) / 2, bit_depth=self.bits).apply(c) \
            if self.stretch else self._disp(c)
        d = cv2.resize(d, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ok, enc = cv2.imencode(".png", d)
        lap = float(cv2.Laplacian(c.astype(np.float32) / ((1 << self.bits) - 1), cv2.CV_32F).var() * 1e6)
        return enc.tobytes(), {"x0": x0, "y0": y0, "size": size, "mean": float(c.mean()), "min": float(c.min()),
                               "max": float(c.max()), "focus": round(lap, 2)}

    # ------------------------------------------------------------------ camera settings (explicit only)
    def apply(self, name: str, value) -> dict:
        self.paused = True
        try:
            with self.lock:
                if hasattr(self.cam, "stop"):
                    self.cam.stop()
                if name == "exposure":
                    self.cam.set_exposure(float(value))
                elif name == "gain":
                    self.cam.set_gain(float(value))
                elif name == "pixel_format":
                    self.cam.set_pixel_format(str(value))
                elif name == "gamma":
                    if hasattr(self.cam, "set_feature"):
                        self.cam.set_feature("Gamma", float(value))
                    else:
                        self.cam.nodes().Gamma.value = float(value)
                else:
                    raise KeyError(name)
            self.info = self._info()
            self.bits = self.info.get("bit_depth") or 8
            return {"ok": True, "applied": {name: value}}
        finally:
            self.paused = False

    # ------------------------------------------------------------------ capture
    def new_session(self, d: dict) -> dict:
        t = CaptureTarget(d.get("bottle", ""), d.get("session", ""), d.get("product_type", ""), d.get("lot", ""),
                          d.get("memo", ""))
        self.writer = CaptureSessionWriter(self.p, t, self.info)
        self.next_angle = float(d.get("start", 0) or 0)
        self.captures = []
        return {"ok": True, "session": self.writer.session_code, "dir": str(self.writer.dir)}

    def capture(self, d: dict) -> dict:
        if self.writer is None:
            return {"ok": False, "error": "start a session first"}
        step = float(d.get("step", 0) or 0)
        n = int(d.get("shots", 1) or 1)
        avg = int(d.get("average", 1) or 1)
        out = []
        last = self.count
        for k in range(n):
            frames = []
            for _ in range(avg):
                img, last = self.wait_new(last)
                frames.append(img)
            img = frames[0] if avg == 1 else np.rint(np.mean(np.stack(frames).astype(np.float64), 0)).astype(
                np.uint16 if self.bits > 8 else np.uint8)
            angle = self.next_angle if step else None
            fid = self.writer.add(img, angle, self.info.get("exposure_us"), self.info.get("gain_db"))
            q = quality(img, self.bits)
            rec = {"frame_id": fid, "index": self.writer.count - 1, "angle": angle, "quality": q,
                   "shape": list(img.shape), "dtype": str(img.dtype)}
            self.captures.append(rec); out.append(rec)
            if step:
                self.next_angle += step
        return {"ok": True, "captured": out, "next_angle": self.next_angle if step else None,
                "count": self.writer.count}

    def thumb(self, idx: int) -> bytes:
        if self.writer is None:
            return b""
        from ..core.image_io import read_image
        f = self.writer.dir / f"frame_{idx:04d}.tif"
        img = read_image(f)
        s = 240 / img.shape[1]
        ok, enc = cv2.imencode(".jpg", self._disp(cv2.resize(img, (240, int(img.shape[0] * s)),
                                                             interpolation=cv2.INTER_AREA)))
        return enc.tobytes()

    def status(self) -> dict:
        with self.cond:
            img = self.frame
        q = quality(img, self.bits) if img is not None else {}
        info = {k: v for k, v in self.info.items() if k != "features"}
        feats = self.info.get("features", {})
        return {"camera": info, "features": {k: feats.get(k) for k in ("Width", "Height", "PixelFormat", "ExposureTime",
                                                                      "Gain", "Gamma", "AcquisitionFrameRate",
                                                                      "DeviceTemperature")},
                "pixel_formats": self.cam.available_pixel_formats() if hasattr(self.cam, "available_pixel_formats") else [],
                "fps": round(self.fps, 2), "frames": self.count,
                "stream_mode": getattr(self.cam, "stream_mode", None),
                "frame_age_ms": int((time.time() - self.frame_t) * 1000) if self.frame_t else None,
                "last_grab_ms": round(self.last_grab_ms, 1), "last_jpeg_ms": round(self.last_jpeg_ms, 1), "quality": q, "error": self.error,
                "stretch": self.stretch, "project": str(self.p.root),
                "session": None if self.writer is None else {"code": self.writer.session_code, "dir": str(self.writer.dir),
                                                             "count": self.writer.count, "next_angle": self.next_angle},
                "captures": self.captures[-50:]}

    def close(self):
        self.running = False
        self.t.join(6)
        with self.lock:
            self.cam.close()


def make_handler(con: Console):
    class H(BaseHTTPRequestHandler):
        # Persistent HTTP/1.1 connections: on this workstation new loopback TCP connections are intermittently
        # reset (measured: 1/3 failures with HTTP/1.0 close-per-request, 0/30 with keep-alive). Every response
        # therefore carries Content-Length and the browser reuses its connections.
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
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
                elif u.path == "/live.jpg":
                    # pull model: the page asks for the next image only after the previous one is shown,
                    # so nothing can queue up (MJPEG streams can be buffered for many seconds by browsers)
                    after = int(q.get("after", -1))
                    if after >= 0:
                        with con.cond:
                            con.cond.wait_for(lambda: con.count > after, 2.0)
                    jpg = con.live_jpeg()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Frame", str(con.jpeg_count))
                    self.send_header("X-Frame-Age-Ms", str(int((time.time() - con.frame_t) * 1000)))
                    self.end_headers()
                    self.wfile.write(jpg)
                elif u.path == "/stream.mjpg":
                    self.close_connection = True   # endless body without Content-Length
                    self.send_response(200)
                    self.send_header("Connection", "close")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = -1
                    while con.running:
                        with con.cond:
                            con.cond.wait_for(lambda: con.count != last, 2.0)
                            last = con.count
                        jpg = con.live_jpeg()
                        if not jpg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                elif u.path == "/crop.png":
                    png, meta = con.crop_jpeg(float(q.get("x", .5)), float(q.get("y", .5)), int(q.get("size", 256)),
                                              int(q.get("scale", 2)))
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("X-Crop", json.dumps(meta))
                    self.send_header("Content-Length", str(len(png)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(png)
                elif u.path == "/api/status":
                    self._json(con.status())
                elif u.path == "/thumb.jpg":
                    self._send(200, con.thumb(int(q.get("i", 0))), "image/jpeg")
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
                if u.path == "/api/session":
                    self._json(con.new_session(d))
                elif u.path == "/api/capture":
                    self._json(con.capture(d))
                elif u.path == "/api/end":
                    con.writer = None
                    self._json({"ok": True})
                elif u.path == "/api/apply":
                    self._json(con.apply(d["name"], d["value"]))
                elif u.path == "/api/view":
                    con.stretch = bool(d.get("stretch", True))
                    self._json({"ok": True})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._json({"ok": False, "error": repr(e)}, 400)
    return H


def serve(project_dir: str, port: int = 8765, mock: bool = False, cti: str | None = None) -> None:
    project = Project.open_or_create(project_dir)
    if mock:
        from .mock import MockCamera, MockRotationStage
        stage = MockRotationStage()
        cam = MockCamera(stage, 1024, 768)
        orig = cam.grab

        def grab(timeout_s=None):   # mock: slowly rotating bottle, ~10 fps
            time.sleep(0.1)
            stage.move_to(stage.position() + 2.0)
            return orig()
        cam.grab = grab
    else:
        from .camera_process import ProcessCamera
        cam = ProcessCamera(cti)       # acquisition in its own process (no GIL contention with the server)
    cam.open()
    print(f"camera stream buffer mode: {getattr(cam, 'stream_mode', None)}", flush=True)
    con = Console(project, cam)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(con))
    srv.daemon_threads = True
    print(f"Camera console on http://127.0.0.1:{port}  (project {project.root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        con.close()
