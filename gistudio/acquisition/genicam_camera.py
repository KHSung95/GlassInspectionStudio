"""GenICam / GenTL camera provider (Harvesters consumer + vendor GenTL producer, e.g. Teledyne Spinnaker).

Policy
------
* Opening, reading features and grabbing frames are allowed. Camera features are only WRITTEN when the
  caller explicitly passes values (exposure, gain, pixel format); defaults leave the camera as configured
  (e.g. in SpinView).
* Frames are returned as raw numpy arrays in the camera's pixel format container (Mono8 -> uint8,
  Mono10/12/16 -> uint16). No scaling, no debayering of mono data.
* The camera can be opened by only one application at a time: close SpinView before connecting.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .interface import CameraProvider

DEFAULT_CTI = r"C:\Program Files\Teledyne\Spinnaker\cti64\vs2015\Spinnaker_GenTL_v140.cti"

INFO_FEATURES = ["DeviceVendorName", "DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion",
                 "Width", "Height", "WidthMax", "HeightMax", "OffsetX", "OffsetY", "PixelFormat", "PixelSize",
                 "AdcBitDepth", "ExposureAuto", "ExposureTime", "GainAuto", "Gain", "GammaEnable", "Gamma",
                 "BlackLevel", "AcquisitionMode", "AcquisitionFrameRateEnable", "AcquisitionFrameRate",
                 "TriggerMode", "TriggerSource", "DeviceLinkThroughputLimit", "GevSCPSPacketSize",
                 "DeviceTemperature", "SensorWidth", "SensorHeight"]

PIXEL_BITS = {"Mono8": 8, "Mono10": 10, "Mono10p": 10, "Mono10Packed": 10, "Mono12": 12, "Mono12p": 12,
              "Mono12Packed": 12, "Mono14": 14, "Mono16": 16}


def _harvester(cti: str):
    from harvesters.core import Harvester
    if not Path(cti).exists():
        raise FileNotFoundError(f"GenTL producer not found: {cti}")
    h = Harvester()
    h.add_file(cti)
    h.update()
    return h


def list_devices(cti: str = DEFAULT_CTI) -> list[dict]:
    h = _harvester(cti)
    try:
        out = []
        for d in h.device_info_list:
            out.append({k: getattr(d, k, None) for k in ("vendor", "model", "serial_number", "id_", "tl_type",
                                                         "user_defined_name", "version", "access_status")})
        return out
    finally:
        h.reset()


def _node_value(nm, name):
    try:
        n = getattr(nm, name)
        return n.value
    except Exception:
        return None


def read_features(cti: str = DEFAULT_CTI, index: int = 0) -> dict:
    h = _harvester(cti)
    try:
        ia = h.create(index)
        try:
            nm = ia.remote_device.node_map
            return {k: _node_value(nm, k) for k in INFO_FEATURES}
        finally:
            ia.destroy()
    finally:
        h.reset()


class GenICamCamera(CameraProvider):
    """Single-frame grabbing from a continuous stream (latest frame), suitable for live view and step capture."""

    def __init__(self, cti: str = DEFAULT_CTI, index: int = 0, serial: str | None = None):
        self.cti, self.index, self.serial = cti, index, serial
        self._h = None
        self._ia = None
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> None:
        if self._ia is not None:
            return
        self._h = _harvester(self.cti)
        if not self._h.device_info_list:
            self._h.reset(); self._h = None
            raise RuntimeError("No GenICam camera found. Check cable/IP and close SpinView (exclusive access).")
        if self.serial:
            self._ia = self._h.create({"serial_number": self.serial})
        else:
            self._ia = self._h.create(self.index)
        # Host-side GenTL stream settings (not camera features): deliver only the newest frame so preview and
        # capture always show the current scene instead of a queue of stale buffers.
        try:
            self._ia.num_buffers = 4
        except Exception:
            pass
        self.stream_mode = None
        try:
            ds_nm = self._ia.data_streams[0].node_map
            ds_nm.StreamBufferHandlingMode.value = "NewestOnly"
            self.stream_mode = ds_nm.StreamBufferHandlingMode.value
        except Exception as e:
            self.stream_mode = f"unchanged ({type(e).__name__}: {e})"

    def close(self) -> None:
        with self._lock:
            if self._ia is not None:
                try:
                    if self._started:
                        self._ia.stop()
                finally:
                    self._ia.destroy()
            self._ia, self._started = None, False
            if self._h is not None:
                self._h.reset(); self._h = None

    @property
    def is_open(self) -> bool:
        return self._ia is not None

    def nodes(self):
        if self._ia is None:
            raise RuntimeError("camera not opened")
        return self._ia.remote_device.node_map

    def info(self) -> dict:
        nm = self.nodes()
        d = {k: _node_value(nm, k) for k in INFO_FEATURES}
        pf = d.get("PixelFormat") or ""
        return {"camera_model": f"{d.get('DeviceVendorName') or ''} {d.get('DeviceModelName') or ''}".strip(),
                "camera_serial": d.get("DeviceSerialNumber"), "pixel_format": pf,
                "bit_depth": PIXEL_BITS.get(pf), "exposure_us": d.get("ExposureTime"), "gain_db": d.get("Gain"),
                "features": d}

    # ------------------------------------------------------------------ explicit writes only
    def _set(self, name: str, value) -> None:
        if self._started:
            raise RuntimeError("stop acquisition before changing camera features")
        getattr(self.nodes(), name).value = value

    def set_exposure(self, us: float) -> None:
        nm = self.nodes()
        if _node_value(nm, "ExposureAuto") not in (None, "Off"):
            self._set("ExposureAuto", "Off")
        self._set("ExposureTime", float(us))

    def set_gain(self, db: float) -> None:
        nm = self.nodes()
        if _node_value(nm, "GainAuto") not in (None, "Off"):
            self._set("GainAuto", "Off")
        self._set("Gain", float(db))

    def set_pixel_format(self, fmt: str) -> None:
        self._set("PixelFormat", fmt)

    def available_pixel_formats(self) -> list[str]:
        try:
            return [s for s in self.nodes().PixelFormat.symbolics if s.startswith("Mono")]
        except Exception:
            return []

    # ------------------------------------------------------------------ grabbing
    def start(self) -> None:
        if not self._started:
            self._ia.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._ia.stop()
            self._started = False

    def grab(self, timeout_s: float = 5.0) -> np.ndarray:
        """Return one new frame (copy) in native bit container."""
        with self._lock:
            if self._ia is None:
                raise RuntimeError("camera not opened")
            self.start()
            with self._ia.fetch(timeout=timeout_s) as buf:
                comp = buf.payload.components[0]
                h, w = comp.height, comp.width
                data = comp.data
                fmt = comp.data_format
                arr = np.array(data, copy=True)
                if arr.size == h * w:
                    arr = arr.reshape(h, w)
                else:  # multi-channel / unexpected: keep first channel only for mono pipeline
                    arr = arr.reshape(h, w, -1)[..., 0]
                if fmt in ("Mono8",) and arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                if fmt not in ("Mono8",) and arr.dtype == np.uint8 and fmt.startswith("Mono") and PIXEL_BITS.get(fmt, 8) > 8:
                    arr = arr.view(np.uint16).reshape(h, w)
                return arr

    def grab_averaged(self, n: int = 1, timeout_s: float = 5.0) -> np.ndarray:
        """Optional temporal averaging (noise reduction for static scenes). n=1 returns the raw frame."""
        if n <= 1:
            return self.grab(timeout_s)
        acc = None
        for _ in range(n):
            f = self.grab(timeout_s).astype(np.float64)
            acc = f if acc is None else acc + f
        return np.rint(acc / n).astype(np.uint16 if acc.max() > 255 else np.uint8)


def capture_session(camera: GenICamCamera, out_dir: str | Path, angles: list[float | None], average: int = 1,
                    progress=None, wait_between_s: float = 0.0) -> Path:
    """Capture one frame per angle entry into a NEW session folder (TIFF + angles.csv + session_meta.json).
    Angles are labels supplied by the operator (manual rotation); None -> unknown angle."""
    import csv
    import json
    from datetime import datetime
    import tifffile
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty (refusing to overwrite captured data)")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, a in enumerate(angles):
        img = camera.grab_averaged(average)
        name = f"frame_{i:04d}.tif"
        tifffile.imwrite(out / name, img)
        rows.append([name, "" if a is None else f"{a:.3f}", i, datetime.now().isoformat()])
        if progress:
            progress(i + 1, len(angles), name)
        if wait_between_s:
            time.sleep(wait_between_s)
    with open(out / "angles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename", "angle", "frame_index", "timestamp"]); w.writerows(rows)
    info = camera.info()
    meta = {"camera_model": info["camera_model"], "bit_depth": info["bit_depth"], "exposure_us": info["exposure_us"],
            "gain_db": info["gain_db"], "captured_at": datetime.now().isoformat(timespec="seconds"),
            "memo": f"serial={info['camera_serial']} pixel_format={info['pixel_format']} average={average}"}
    (out / "session_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    (out / "camera_features.json").write_text(json.dumps(info["features"], indent=1, default=str), encoding="utf-8")
    return out
