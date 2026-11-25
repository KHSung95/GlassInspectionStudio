"""Glass Inspection Studio — unified local web app (project / capture / labelling / analysis / report).

    python -m gistudio.cli studio [--port 8800] [--project C:\\dev\\gis_real2d]

* Binds to 127.0.0.1 only; images never leave this machine.
* One project is open at a time; switching closes the labelling state and the capture session.
* The camera is connected only on request (capture tab) and runs in its own high-priority process.
* Heavy work (import, detector runs, sweeps, reports) runs as separate BELOW-NORMAL priority processes ("jobs")
  so capture and the UI stay responsive. Each job writes a log under <project>/logs.
* The project database is backed up when a project is opened (last 10 kept under <project>/backups).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..core.project import Project
from .web_label import LabelServer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDIO_PAGE = HERE / "web_studio.html"
LABEL_PAGE = HERE / "web_label.html"
CONSOLE_PAGE = REPO / "gistudio" / "acquisition" / "web_console.html"
SETTINGS_FILE = Path.home() / ".gistudio" / "studio.json"
DEFAULT_CAMERA = {"packet_size": 8192, "fps": 5.0, "cti": None}
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"recent": []}


def _save_settings(s: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def backup_db(project: Project, keep: int = 10) -> Path:
    """Consistent online copy of studio.db (sqlite backup API); keeps the newest `keep` copies."""
    d = project.root / "backups"
    d.mkdir(exist_ok=True)
    dst = d / f"studio_{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    src = sqlite3.connect(str(project.root / Project.DB_NAME))
    out = sqlite3.connect(str(dst))
    with out:
        src.backup(out)
    out.close(); src.close()
    for old in sorted(d.glob("studio_*.db"))[:-keep]:
        old.unlink(missing_ok=True)
    return dst


class Jobs:
    """Background processes with logs. Never blocks the server."""

    def __init__(self):
        self.items: list[dict] = []
        self.lock = threading.Lock()

    def start(self, title: str, args: list[str], project_root: Path, on_done=None) -> dict:
        logs = project_root / "logs"
        logs.mkdir(exist_ok=True)
        jid = len(self.items) + 1
        log = logs / f"job_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{jid}.log"
        fh = open(log, "w", encoding="utf-8")
        fh.write(f"$ {' '.join(args)}\n"); fh.flush()
        flags = (BELOW_NORMAL | NO_WINDOW) if os.name == "nt" else 0
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(args, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, creationflags=flags, env=env)
        job = {"id": jid, "title": title, "log": str(log), "status": "running", "started": time.strftime("%H:%M:%S"),
               "ended": None, "returncode": None, "_p": p, "_fh": fh, "_done": on_done}
        with self.lock:
            self.items.append(job)
        threading.Thread(target=self._wait, args=(job,), daemon=True).start()
        return self.public(job)

    def _wait(self, job):
        rc = job["_p"].wait()
        job["_fh"].close()
        job.update(status="done" if rc == 0 else "failed", returncode=rc, ended=time.strftime("%H:%M:%S"))
        if job["_done"]:
            try:
                job["_done"](job)
            except Exception:
                pass

    @staticmethod
    def public(job) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def list(self) -> list[dict]:
        out = []
        for j in reversed(self.items[-30:]):
            d = self.public(j)
            try:
                d["tail"] = Path(j["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            except Exception:
                d["tail"] = []
            out.append(d)
        return out

    def cancel(self, jid: int) -> bool:
        for j in self.items:
            if j["id"] == jid and j["status"] == "running":
                j["_p"].terminate(); return True
        return False


class Studio:
    def __init__(self):
        self.settings = _load_settings()
        self.project: Project | None = None
        self.labels: LabelServer | None = None
        self.console = None
        self.cam_lock = threading.Lock()
        self.cam_error = ""
        self.jobs = Jobs()
        self.last_backup = None

    # ------------------------------------------------------------------ projects
    def open_project(self, path: str, create: bool = False) -> dict:
        root = Path(path).expanduser()
        p = Project.create(root) if create else Project(root)
        if self.console is not None:
            self.console.writer = None
            self.console.p = p
        self.project = p
        self.labels = LabelServer(p)
        self.last_backup = str(backup_db(p))
        rec = [str(p.root)] + [r for r in self.settings.get("recent", []) if r != str(p.root)]
        self.settings["recent"] = rec[:10]
        _save_settings(self.settings)
        return self.project_info()

    def project_settings(self) -> dict:
        f = self.project.root / "studio_settings.json"
        try:
            return {**{"camera": dict(DEFAULT_CAMERA)}, **json.loads(f.read_text(encoding="utf-8"))}
        except Exception:
            return {"camera": dict(DEFAULT_CAMERA)}

    def save_project_settings(self, d: dict) -> dict:
        s = self.project_settings()
        s.update(d)
        (self.project.root / "studio_settings.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        return s

    def project_info(self) -> dict:
        if self.project is None:
            return {"open": False, "recent": self.recent()}
        p = self.project
        bottles = []
        for b in p.bottles():
            for s in p.sessions(b["id"]):
                n = p.db.one("""SELECT COUNT(*), SUM(review_status='reviewed') FROM frames WHERE session_id=?""", (s["id"],))
                la = p.db.one("""SELECT COUNT(*), SUM(a.label_state='positive') FROM annotations a JOIN frames f
                                 ON a.frame_id=f.id WHERE f.session_id=? AND a.deleted=0""", (s["id"],))
                bottles.append({"bottle": b["bottle_code"], "product_type": b["product_type"], "verdict": b["verdict"],
                                "session": s["session_code"], "frames": n[0], "reviewed": n[1] or 0,
                                "labels": la[0], "defects": la[1] or 0, "camera": s["camera_model"],
                                "source": s["source_dir"]})
        per_class = [dict(r) for r in p.db.query(
            """SELECT class_name, label_state, COUNT(*) AS n FROM annotations WHERE deleted=0
               GROUP BY class_name, label_state ORDER BY class_name""")]
        return {"open": True, "root": str(p.root), "name": p.name, "stats": p.stats(), "sessions": bottles,
                "labels_by_class": per_class, "settings": self.project_settings(), "last_backup": self.last_backup,
                "recent": self.recent()}

    def recent(self) -> list[str]:
        return [r for r in self.settings.get("recent", []) if (Path(r) / Project.DB_NAME).exists()]

    def require_project(self):
        if self.project is None:
            raise RuntimeError("먼저 프로젝트를 여세요.")
        return self.project

    # ------------------------------------------------------------------ camera
    def camera_connect(self) -> dict:
        p = self.require_project()
        with self.cam_lock:
            if self.console is not None:
                return {"ok": True, "already": True}
            from ..acquisition.camera_process import ProcessCamera
            from ..acquisition.web_console import Console
            c = self.project_settings()["camera"]
            startup = {}
            if c.get("packet_size"):
                startup["GevSCPSPacketSize"] = int(c["packet_size"])
            if c.get("fps"):
                startup["AcquisitionFrameRate"] = float(c["fps"])
            cam = ProcessCamera(c.get("cti"), startup=startup)
            try:
                cam.open()
            except Exception as e:
                self.cam_error = repr(e)
                msg = repr(e)
                if "AccessDenied" in msg or "not allowed" in msg:
                    msg += " — 다른 프로그램(SpinView 등)이 카메라를 쓰고 있는지 확인하세요."
                raise RuntimeError(msg)
            self.console = Console(p, cam)
            self.cam_error = ""
            return {"ok": True, "startup": getattr(cam, "applied", {}), "priority": getattr(cam, "priority", None)}

    def camera_disconnect(self) -> dict:
        with self.cam_lock:
            if self.console is not None:
                c, self.console = self.console, None
                c.close()
        return {"ok": True}

    # ------------------------------------------------------------------ jobs
    def py(self) -> str:
        return sys.executable

    def job_import(self, d: dict) -> dict:
        p = self.require_project()
        src = Path(d["path"].strip().strip('"'))
        if not src.exists():
            raise FileNotFoundError(src)
        if src.suffix.lower() == ".zip":
            dest = p.root / "imported" / src.stem
            if dest.exists() and any(dest.iterdir()):
                raise FileExistsError(f"이미 가져온 zip입니다: {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src) as z:
                for m in z.infolist():
                    target = (dest / m.filename).resolve()
                    if not str(target).startswith(str(dest.resolve())):
                        raise ValueError(f"unsafe path in zip: {m.filename}")
                z.extractall(dest)
            subdirs = [x for x in dest.iterdir() if x.is_dir()]
            src = subdirs[0] if len(subdirs) == 1 and not any(x.is_file() for x in dest.iterdir()) else dest
        args = [self.py(), "-m", "gistudio.cli", "import", str(p.root), str(src), "--layout", d.get("layout", "flat"),
                "--dataset", d.get("dataset") or "default"]
        if d.get("bottle"):
            args += ["--bottle", d["bottle"]]
        if d.get("angle_step"):
            args += ["--angle-step", str(d["angle_step"])]
        return self.jobs.start(f"가져오기: {src.name}", args, p.root)

    def job_sweep(self, d: dict) -> dict:
        p = self.require_project()
        groups = ",".join(d.get("groups") or [])
        args = [self.py(), str(REPO / "tools" / "per_class_sweep.py"), str(p.root),
                str(p.root / "exports" / "per_class"), "--combos", str(int(d.get("combos", 8)))]
        if groups:
            args += ["--only", groups]
        return self.jobs.start(f"결함별 sweep ({groups or '전체'})", args, p.root)

    def job_detect(self, d: dict) -> dict:
        p = self.require_project()
        args = [self.py(), "-m", "gistudio.cli", "run", str(p.root), "--algorithm", d["algorithm"],
                "--frames", d.get("frames", "all"), "--params", json.dumps(d.get("params", {})), "--no-maps"]
        return self.jobs.start(f"검출기 실행: {d['algorithm']}", args, p.root)

    def job_report(self, d: dict) -> dict:
        p = self.require_project()
        out = p.root / "exports" / f"report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
        args = [self.py(), str(REPO / "tools" / "studio_report.py"), str(p.root), str(out)]
        if d.get("title"):
            args += ["--title", d["title"]]
        return self.jobs.start("보고서 생성", args, p.root)

    def analysis_results(self) -> dict:
        p = self.require_project()
        f = p.root / "exports" / "per_class" / "per_class_summary.json"
        summary = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        runs = [{k: r[k] for k in ("id", "kind", "algorithm", "status", "created_at", "notes")}
                for r in p.runs() if r["kind"] in ("classical", "zeroshot", "infer")][:20]
        for r in runs:
            r["candidates"] = p.db.one("SELECT COUNT(*) FROM candidates WHERE run_id=?", (r["id"],))[0]
        return {"per_class": summary, "per_class_time": time.ctime(f.stat().st_mtime) if f.exists() else None,
                "runs": runs}

    def reports(self) -> list[dict]:
        p = self.require_project()
        d = p.root / "exports"
        out = []
        for f in sorted(d.glob("report_*.html"), reverse=True) + sorted(d.glob("**/*REPORT*.md"), reverse=True):
            out.append({"name": f.name, "rel": str(f.relative_to(p.root)).replace("\\", "/"),
                        "time": time.ctime(f.stat().st_mtime)})
        return out


def make_handler(st: Studio):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        # ------------------------------------------------------------------ GET
        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = u.path
            try:
                if path == "/":
                    return self._send(200, STUDIO_PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/label":
                    return self._send(200, LABEL_PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/camera":
                    return self._send(200, CONSOLE_PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/studio/project":
                    return self._json(st.project_info())
                if path == "/studio/jobs":
                    return self._json(st.jobs.list())
                if path == "/studio/analysis":
                    return self._json(st.analysis_results())
                if path == "/studio/reports":
                    return self._json(st.reports())
                if path == "/studio/algorithms":
                    from ..algorithms.base import registry
                    return self._json([{"id": k, "name": v.display_name, "family": v.family,
                                        "default_class": v.default_class}
                                       for k, v in registry().items() if v.family == "classical"])
                if path.startswith("/files/"):
                    p = st.require_project()
                    f = (p.root / unquote(path[len("/files/"):])).resolve()
                    if not str(f).startswith(str(p.root.resolve())) or not f.is_file():
                        return self._send(404, b"not found", "text/plain")
                    ctype = {".html": "text/html; charset=utf-8", ".md": "text/plain; charset=utf-8",
                             ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}.get(
                        f.suffix.lower(), "application/octet-stream")
                    return self._send(200, f.read_bytes(), ctype)
                # ---------- labelling API
                if path == "/api/frames":
                    return self._json(self.labels().frames())
                if path == "/api/classes":
                    return self._json(st.require_project().classes())
                if m := re.fullmatch(r"/api/frame/(\d+)/image\.jpg", path):
                    return self._send(200, self.labels().image(int(m[1]), q.get("mode", "stretch")), "image/jpeg")
                if m := re.fullmatch(r"/api/frame/(\d+)/anns", path):
                    return self._json(self.labels().anns(int(m[1])))
                # ---------- camera API
                con = st.console
                if path == "/api/status":
                    if con is None:
                        return self._json({"connected": False, "error": st.cam_error,
                                           "settings": st.project_settings()["camera"] if st.project else None})
                    s = con.status(); s["connected"] = True
                    return self._json(s)
                if con is None and path in ("/live.jpg", "/crop.png", "/thumb.jpg", "/api/stream_stats"):
                    return self._send(409, b"camera not connected", "text/plain")
                if path == "/api/stream_stats":
                    return self._json(con.cam.stats() if hasattr(con.cam, "stats") else {})
                if path == "/live.jpg":
                    after = int(q.get("after", -1))
                    if after >= 0:
                        with con.cond:
                            con.cond.wait_for(lambda: con.count > after, 2.0)
                    jpg = con.live_jpeg()
                    return self._send(200, jpg, "image/jpeg", {"X-Frame": str(con.jpeg_count),
                                                               "X-Frame-Age-Ms": str(int((time.time() - con.frame_t) * 1000))})
                if path == "/crop.png":
                    png, meta = con.crop_jpeg(float(q.get("x", .5)), float(q.get("y", .5)), int(q.get("size", 256)),
                                              int(q.get("scale", 2)))
                    return self._send(200, png, "image/png", {"X-Crop": json.dumps(meta)})
                if path == "/thumb.jpg":
                    return self._send(200, con.thumb(int(q.get("i", 0))), "image/jpeg")
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
            except Exception as e:
                self._json({"ok": False, "error": str(e), "trace": traceback.format_exc()}, 500)

        def labels(self) -> LabelServer:
            st.require_project()
            return st.labels

        # ------------------------------------------------------------------ POST
        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0) or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
            path = u.path
            try:
                if path == "/studio/open":
                    return self._json({"ok": True, "project": st.open_project(d["path"], bool(d.get("create")))})
                if path == "/studio/settings":
                    st.require_project()
                    return self._json({"ok": True, "settings": st.save_project_settings(d)})
                if path == "/studio/backup":
                    p = st.require_project(); st.last_backup = str(backup_db(p))
                    return self._json({"ok": True, "backup": st.last_backup})
                if path == "/studio/export_labels":
                    from ..core.export import export_project_annotations
                    p = st.require_project()
                    out = p.root / "exports" / f"annotations_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
                    n_ann = export_project_annotations(p, str(out))
                    return self._json({"ok": True, "file": str(out), "annotations": n_ann})
                if path == "/studio/import":
                    return self._json({"ok": True, "job": st.job_import(d)})
                if path == "/studio/sweep":
                    return self._json({"ok": True, "job": st.job_sweep(d)})
                if path == "/studio/detect":
                    return self._json({"ok": True, "job": st.job_detect(d)})
                if path == "/studio/report":
                    return self._json({"ok": True, "job": st.job_report(d)})
                if path == "/studio/job_cancel":
                    return self._json({"ok": st.jobs.cancel(int(d["id"]))})
                if path == "/studio/camera/connect":
                    return self._json(st.camera_connect())
                if path == "/studio/camera/disconnect":
                    return self._json(st.camera_disconnect())
                # ---------- labelling API
                if m := re.fullmatch(r"/api/frame/(\d+)/anns", path):
                    return self._json({"ok": True, "ann": self.labels().add(int(m[1]), d)})
                if m := re.fullmatch(r"/api/frame/(\d+)/review", path):
                    st.require_project().set_frame_review_status(int(m[1]), d["status"]); return self._json({"ok": True})
                if m := re.fullmatch(r"/api/ann/(\d+)/delete", path):
                    self.labels().delete(int(m[1])); return self._json({"ok": True})
                if m := re.fullmatch(r"/api/ann/(\d+)", path):
                    return self._json({"ok": True, "ann": self.labels().update(int(m[1]), d)})
                # ---------- camera API
                con = st.console
                if path in ("/api/session", "/api/capture", "/api/end", "/api/apply", "/api/view") and con is None:
                    return self._json({"ok": False, "error": "카메라가 연결되지 않았습니다."}, 409)
                if path == "/api/session":
                    return self._json(con.new_session(d))
                if path == "/api/capture":
                    return self._json(con.capture(d))
                if path == "/api/end":
                    con.writer = None; return self._json({"ok": True})
                if path == "/api/apply":
                    return self._json(con.apply(d["name"], d["value"]))
                if path == "/api/view":
                    con.stretch = bool(d.get("stretch", True)); return self._json({"ok": True})
                return self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)
    return H


def serve(port: int = 8800, project: str | None = None, open_browser: bool = False) -> None:
    st = Studio()
    if project:
        st.open_project(project)
    elif st.recent():
        try:
            st.open_project(st.recent()[0])
        except Exception:
            pass
    http = ThreadingHTTPServer(("127.0.0.1", port), make_handler(st))
    http.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"Glass Inspection Studio on {url}  (project: {st.project.root if st.project else '-'})", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        st.camera_disconnect()
        http.server_close()
