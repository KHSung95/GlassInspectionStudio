"""Run the GenICam camera in its own process; the newest frame is published through shared memory.

Why: inside a busy Python process (HTTP server, JPEG encoding, GUI) the Harvesters acquisition thread competes
for the GIL and frame delivery stalls for seconds. A dedicated process only grabs, so the newest frame is always
fresh; consumers copy it from shared memory.

The proxy implements the same methods the console/GUI use (open/close/info/grab/set_*), so it is a drop-in.
"""
from __future__ import annotations

import multiprocessing as mp
import struct
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

HDR = struct.Struct("<qdiii")          # count, timestamp, height, width, itemsize
HDR_SIZE = 64


def _worker(cti: str, shm_name: str, cap: int, lock, cmd, ready):
    from .genicam_camera import GenICamCamera
    shm = shared_memory.SharedMemory(name=shm_name)
    cam = GenICamCamera(cti)
    try:
        cam.open()
        ready.send({"ok": True, "info": cam.info(), "stream_mode": cam.stream_mode,
                    "pixel_formats": cam.available_pixel_formats()})
    except Exception as e:
        ready.send({"ok": False, "error": repr(e)})
        shm.close()
        return
    count = 0
    running = True
    while running:
        # commands (settings, info, stop) are handled between frames
        while cmd.poll():
            name, arg = cmd.recv()
            try:
                if name == "stop":
                    running = False; res = None
                elif name == "info":
                    res = cam.info()
                else:
                    cam.stop()
                    if name == "exposure":
                        cam.set_exposure(float(arg))
                    elif name == "gain":
                        cam.set_gain(float(arg))
                    elif name == "pixel_format":
                        cam.set_pixel_format(str(arg))
                    elif name == "feature":
                        setattr_name, value = arg
                        getattr(cam.nodes(), setattr_name).value = value
                    res = cam.info()
                cmd.send({"ok": True, "result": res})
            except Exception as e:
                cmd.send({"ok": False, "error": repr(e)})
        if not running:
            break
        try:
            img = cam.grab(timeout_s=2.0)
        except Exception:
            time.sleep(0.05)
            continue
        n = img.nbytes
        if n + HDR_SIZE > cap:
            continue
        count += 1
        with lock:
            shm.buf[HDR_SIZE:HDR_SIZE + n] = img.tobytes()
            shm.buf[:HDR.size] = HDR.pack(count, time.time(), img.shape[0], img.shape[1], img.itemsize)
    try:
        cam.close()
    finally:
        shm.close()


class ProcessCamera:
    def __init__(self, cti: str | None = None, max_bytes: int = 4128 * 3008 * 2):
        from .genicam_camera import DEFAULT_CTI
        self.cti = cti or DEFAULT_CTI
        self.cap = max_bytes + HDR_SIZE
        self.shm = None
        self.proc = None
        self._info = {}
        self.stream_mode = None
        self._formats = []
        self._last = 0
        self._ctx = mp.get_context("spawn")
        self.lock = self._ctx.Lock()

    def open(self) -> None:
        self.shm = shared_memory.SharedMemory(create=True, size=self.cap)
        self.shm.buf[:HDR.size] = HDR.pack(0, 0.0, 0, 0, 1)
        self.cmd, child_cmd = self._ctx.Pipe()
        ready_r, ready_w = self._ctx.Pipe(duplex=False)
        self.proc = self._ctx.Process(target=_worker, args=(self.cti, self.shm.name, self.cap, self.lock, child_cmd,
                                                            ready_w), daemon=True)
        self.proc.start()
        if not ready_r.poll(60):
            self.close(); raise TimeoutError("camera process did not start")
        msg = ready_r.recv()
        if not msg["ok"]:
            self.close(); raise RuntimeError(msg["error"])
        self._info, self.stream_mode, self._formats = msg["info"], msg["stream_mode"], msg["pixel_formats"]

    def _call(self, name, arg=None, timeout=30.0):
        self.cmd.send((name, arg))
        if not self.cmd.poll(timeout):
            raise TimeoutError(f"camera process did not answer '{name}'")
        r = self.cmd.recv()
        if not r["ok"]:
            raise RuntimeError(r["error"])
        return r["result"]

    def info(self) -> dict:
        return self._info

    def refresh_info(self) -> dict:
        self._info = self._call("info")
        return self._info

    def set_exposure(self, us):
        self._info = self._call("exposure", us)

    def set_gain(self, db):
        self._info = self._call("gain", db)

    def set_pixel_format(self, fmt):
        self._info = self._call("pixel_format", fmt)

    def set_feature(self, name, value):
        self._info = self._call("feature", (name, value))

    def available_pixel_formats(self):
        return list(self._formats)

    def stop(self):
        pass  # the worker stops/restarts acquisition itself around feature writes

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        with self.lock:
            count, ts, h, w, isz = HDR.unpack(bytes(self.shm.buf[:HDR.size]))
            if count == 0:
                return None, 0, 0.0
            dt = np.uint8 if isz == 1 else np.uint16
            img = np.frombuffer(self.shm.buf, dtype=dt, count=h * w, offset=HDR_SIZE).reshape(h, w).copy()
        return img, count, ts

    def grab(self, timeout_s: float = 5.0) -> np.ndarray:
        """Newest frame that is newer than the previously returned one."""
        t0 = time.time()
        while True:
            with self.lock:
                count = HDR.unpack(bytes(self.shm.buf[:HDR.size]))[0]
            if count > self._last:
                img, c, _ = self.latest()
                self._last = c
                return img
            if time.time() - t0 > timeout_s:
                raise TimeoutError("no new frame")
            if self.proc is not None and not self.proc.is_alive():
                raise RuntimeError("camera process exited")
            time.sleep(0.004)

    def close(self) -> None:
        try:
            if self.proc is not None and self.proc.is_alive():
                try:
                    self._call("stop", timeout=10)
                except Exception:
                    pass
                self.proc.join(10)
                if self.proc.is_alive():
                    self.proc.terminate()
        finally:
            self.proc = None
            if self.shm is not None:
                try:
                    self.shm.close(); self.shm.unlink()
                except Exception:
                    pass
                self.shm = None
