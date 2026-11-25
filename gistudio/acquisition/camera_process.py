"""Run the GenICam camera in its own process; the newest frame is published through shared memory.

Why a separate process
----------------------
Inside a busy Python process (HTTP server, JPEG encoding, GUI) the Harvesters acquisition thread competes for
the GIL and frame delivery stalls. A dedicated, higher-priority process only grabs; consumers copy the newest
frame from shared memory.

Robustness
----------
GigE streams can die under load (packet loss, heartbeat timeouts) and native calls may then block forever.
The parent therefore runs a watchdog: if no new frame arrives for `watchdog_s`, the worker process is killed
and a fresh one re-opens the camera. Restarts and GenTL stream statistics are exposed via `stats()`.
"""
from __future__ import annotations

import multiprocessing as mp
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np

HDR = struct.Struct("<qdiii")          # count, timestamp, height, width, itemsize
HDR_SIZE = 64
STREAM_STATS = ["StreamDeliveredFrameCount", "StreamLostFrameCount", "StreamIncompleteFrameCount",
                "StreamDroppedFrameCount", "StreamUnderrunCount", "StreamFailedBufferCount",
                "StreamReceivedPacketCount", "StreamMissedPacketCount", "StreamPacketResendRequestCount",
                "StreamPacketResendRequestSuccessCount", "StreamBufferCountResult", "StreamBufferHandlingMode"]


def _raise_priority() -> str:
    """Keep GigE packet handling / heartbeats on time when other work loads the CPU (Windows)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k32.GetPriorityClass.restype = wintypes.DWORD
        h = k32.GetCurrentProcess()
        HIGH, ABOVE = 0x00000080, 0x00008000
        ok = k32.SetPriorityClass(h, HIGH) or k32.SetPriorityClass(h, ABOVE)
        return {HIGH: "high", ABOVE: "above_normal"}.get(k32.GetPriorityClass(h), f"0x{k32.GetPriorityClass(h):x}") \
            if ok else f"unchanged (error {ctypes.get_last_error()})"
    except Exception as e:
        return f"unchanged ({e!r})"


def _stream_stats(cam) -> dict:
    out = {}
    try:
        nm = cam._ia.data_streams[0].node_map
    except Exception:
        return out
    for n in STREAM_STATS:
        try:
            out[n] = getattr(nm, n).value
        except Exception:
            pass
    return out


def _worker(cti: str, shm_name: str, cap: int, lock, cmd, ready, startup: dict | None = None):
    from .genicam_camera import GenICamCamera
    prio = _raise_priority()
    shm = shared_memory.SharedMemory(name=shm_name)
    cam = GenICamCamera(cti)
    try:
        cam.open()
        applied = {}
        # explicit, operator-requested feature writes (e.g. GevSCPSPacketSize, AcquisitionFrameRate) before streaming
        for fname, value in (startup or {}).items():
            node = getattr(cam.nodes(), fname)
            node.value = value
            applied[fname] = node.value
        ready.send({"ok": True, "info": cam.info(), "stream_mode": cam.stream_mode, "priority": prio,
                    "pixel_formats": cam.available_pixel_formats(), "applied": applied})
    except Exception as e:
        ready.send({"ok": False, "error": repr(e)})
        shm.close()
        return
    with lock:
        count = HDR.unpack(bytes(shm.buf[:HDR.size]))[0]   # continue numbering across worker restarts
    running = True
    while running:
        while cmd.poll():
            name, arg = cmd.recv()
            try:
                if name == "stop":
                    running = False; res = None
                elif name == "info":
                    res = cam.info()
                elif name == "stats":
                    res = _stream_stats(cam)
                else:
                    cam.stop()
                    if name == "exposure":
                        cam.set_exposure(float(arg))
                    elif name == "gain":
                        cam.set_gain(float(arg))
                    elif name == "pixel_format":
                        cam.set_pixel_format(str(arg))
                    elif name == "feature":
                        fname, value = arg
                        getattr(cam.nodes(), fname).value = value
                    res = cam.info()
                cmd.send({"ok": True, "result": res})
            except Exception as e:
                cmd.send({"ok": False, "error": repr(e)})
        if not running:
            break
        try:
            img = cam.grab(timeout_s=2.0)
        except Exception:
            continue   # the parent watchdog restarts this process if the stream stays dead
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
    def __init__(self, cti: str | None = None, max_bytes: int = 4128 * 3008 * 2, watchdog_s: float = 6.0,
                 startup: dict | None = None):
        from .genicam_camera import DEFAULT_CTI
        self.cti = cti or DEFAULT_CTI
        self.startup = dict(startup or {})
        self.applied = {}
        self.cap = max_bytes + HDR_SIZE
        self.watchdog_s = watchdog_s
        self.shm = None
        self.proc = None
        self._info = {}
        self.stream_mode = None
        self.priority = None
        self._formats = []
        self._last = 0
        self._ctx = mp.get_context("spawn")
        self.lock = self._ctx.Lock()
        self._cmd_lock = threading.Lock()
        self._restarts = 0
        self._restart_log: list[str] = []
        self._closing = False
        self._wd = None

    # ------------------------------------------------------------------ worker lifecycle
    def _spawn(self, timeout: float = 60.0) -> None:
        self.cmd, child_cmd = self._ctx.Pipe()
        ready_r, ready_w = self._ctx.Pipe(duplex=False)
        self.proc = self._ctx.Process(target=_worker, args=(self.cti, self.shm.name, self.cap, self.lock, child_cmd,
                                                            ready_w, self.startup), daemon=True)
        self.proc.start()
        if not ready_r.poll(timeout):
            self._kill()
            raise TimeoutError("camera process did not start")
        msg = ready_r.recv()
        if not msg["ok"]:
            self._kill()
            raise RuntimeError(msg["error"])
        self._info, self.stream_mode, self._formats = msg["info"], msg["stream_mode"], msg["pixel_formats"]
        self.priority = msg.get("priority")
        self.applied = msg.get("applied", {})

    def _kill(self) -> None:
        if self.proc is not None:
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(5)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join(5)
            self.proc = None

    def open(self) -> None:
        self.shm = shared_memory.SharedMemory(create=True, size=self.cap)
        self.shm.buf[:HDR.size] = HDR.pack(0, 0.0, 0, 0, 1)
        self._spawn()
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    def _header(self):
        with self.lock:
            return HDR.unpack(bytes(self.shm.buf[:HDR.size]))

    def _watchdog(self) -> None:
        last_count, last_change = -1, time.time()
        while not self._closing:
            time.sleep(0.5)
            if self.shm is None:
                break
            count, ts = self._header()[:2]
            if count != last_count:
                last_count, last_change = count, time.time()
                continue
            if time.time() - last_change < self.watchdog_s or self._cmd_lock.locked():
                continue
            # no frame for watchdog_s: kill the (possibly blocked) worker and start a fresh one
            with self._cmd_lock:
                if self._closing:
                    break
                self._restarts += 1
                self._restart_log.append(time.strftime("%H:%M:%S") + " no frame for %.0fs -> restart" %
                                         (time.time() - last_change))
                self._kill()
                for attempt in range(10):
                    if self._closing:
                        return
                    try:
                        time.sleep(1.0 + attempt)   # let the camera release the control channel (heartbeat)
                        self._spawn()
                        break
                    except Exception as e:
                        self._restart_log.append(time.strftime("%H:%M:%S") + f" reopen failed: {e!r}"[:160])
                last_change = time.time()

    def _call(self, name, arg=None, timeout=30.0):
        with self._cmd_lock:
            self.cmd.send((name, arg))
            if not self.cmd.poll(timeout):
                raise TimeoutError(f"camera process did not answer '{name}'")
            r = self.cmd.recv()
        if not r["ok"]:
            raise RuntimeError(r["error"])
        return r["result"]

    # ------------------------------------------------------------------ camera API (drop-in)
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

    def reconnects(self) -> int:
        return self._restarts

    def stats(self) -> dict:
        try:
            s = self._call("stats", timeout=3.0)
        except Exception as e:
            s = {"error": repr(e)}
        return {"worker_restarts": self._restarts, "restart_log": self._restart_log[-5:],
                "priority": self.priority, "startup_applied": self.applied, "stream": s}

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
            count = self._header()[0]
            if count > self._last:
                img, c, _ = self.latest()
                self._last = c
                return img
            if time.time() - t0 > timeout_s:
                raise TimeoutError("no new frame")
            time.sleep(0.004)

    def close(self) -> None:
        self._closing = True
        try:
            if self.proc is not None and self.proc.is_alive():
                try:
                    self._call("stop", timeout=10)
                except Exception:
                    pass
                self.proc.join(10)
            self._kill()
        finally:
            if self.shm is not None:
                try:
                    self.shm.close(); self.shm.unlink()
                except Exception:
                    pass
                self.shm = None
