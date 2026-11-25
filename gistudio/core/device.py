"""Compute device detection with CPU fallback."""
from __future__ import annotations

import platform
import shutil
import subprocess


def detect() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": None,
            "cuda_available": False, "cuda_devices": [], "xpu_available": False, "nvidia_smi": None}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            info["nvidia_smi"] = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total",
                                                 "--format=csv,noheader"], capture_output=True, text=True,
                                                timeout=10).stdout.strip()
        except Exception as e:  # pragma: no cover
            info["nvidia_smi"] = f"error: {e}"
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["xpu_available"] = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    return info


def select_device(pref: str = "auto"):
    """'auto' -> cuda if available, else cpu. Explicit 'cuda:1', 'xpu', 'cpu' also accepted."""
    import torch
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (no NVIDIA GPU / driver or CPU-only PyTorch build)")
    if pref.startswith("xpu") and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        raise RuntimeError("XPU requested but not available")
    return torch.device(pref)


def memory_report(device) -> dict:
    import torch
    if device.type == "cuda":
        return {"allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20}
    return {}
