"""Shared helpers for vision-language zero-shot models: model cache location, tiled CLIP inference.

Models are downloaded only from their official distribution (open_clip / Hugging Face hub) into
<repo>/models/hf_cache. No images are ever sent to any external service: inference is fully local.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from ...core.tiling import Stitcher, make_tiles
from ..base import Candidate
from ..classical import ops

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "models"
MODEL_REGISTRY_FILE = MODEL_DIR / "model_registry.json"


_patched = False


def configure_model_cache() -> None:
    """Keep every pretrained weight inside <repo>/models (reproducible, no hidden user-profile caches)."""
    global _patched
    os.environ.setdefault("HF_HOME", str(MODEL_DIR / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch_cache"))
    if _patched:
        return
    try:
        # open_clip downloads GitHub-release weights to ~/.cache/clip by default; redirect to the project folder.
        import open_clip.pretrained as ocp
        orig = ocp.download_pretrained_from_url

        def _local(url, cache_dir=None, _orig=orig):
            return _orig(url, cache_dir=cache_dir or str(MODEL_DIR / "clip_cache"))
        ocp.download_pretrained_from_url = _local
        _patched = True
    except Exception:
        pass


def model_registry() -> dict:
    if MODEL_REGISTRY_FILE.exists():
        return json.loads(MODEL_REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def tile_to_clip_tensor(tiles: list[np.ndarray], size: int, bit_depth: int | None, contrast: str):
    """Mono tiles (native resolution, raw values) -> normalised (B, 3, size, size) float tensor.

    contrast='fixed'      : value / (2^bits - 1)  (keeps absolute brightness; recommended)
    contrast='percentile' : per-tile 0.5-99.5% stretch (may hide low-contrast intensity defects!)
    """
    import torch
    out = []
    for t in tiles:
        x = t.astype(np.float32)
        if contrast == "percentile":
            lo, hi = np.percentile(x, (0.5, 99.5))
            x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            mx = float((1 << bit_depth) - 1) if bit_depth else (float(np.iinfo(t.dtype).max)
                                                                 if np.issubdtype(t.dtype, np.integer) else 1.0)
            x = np.clip(x / mx, 0, 1)
        interp = cv2.INTER_AREA if x.shape[0] > size else cv2.INTER_CUBIC
        x = cv2.resize(x, (size, size), interpolation=interp)
        out.append(np.stack([x, x, x], 0))
    arr = torch.from_numpy(np.stack(out, 0))
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (arr - mean) / std


def tiled_map(image: np.ndarray, tile: int, overlap: int, model_size: int, batch: int, fn, bit_depth, contrast,
              progress=None) -> np.ndarray:
    """Run fn(tensor (B,3,S,S)) -> np (B, h, w) maps over native-resolution tiles and stitch back (max)."""
    h, w = image.shape
    tiles = make_tiles(h, w, tile, overlap)
    st = Stitcher(h, w, 1, mode="max")
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        crops = [image[t.y0:t.y1, t.x0:t.x1] for t in chunk]
        maps = fn(tile_to_clip_tensor(crops, model_size, bit_depth, contrast))
        for t, m in zip(chunk, maps):
            m = cv2.resize(m.astype(np.float32), (t.w, t.h), interpolation=cv2.INTER_LINEAR)
            st.add(t, m)
        if progress:
            progress(min(i + batch, len(tiles)), len(tiles), "tiles")
    return st.result()[0]


def resolution_warnings(tile: int, model_size: int, feature_stride: int) -> tuple[list[str], dict]:
    scale = model_size / float(tile)
    eff = feature_stride / scale
    info = {"tile_px": tile, "model_input_px": model_size, "resize_factor": scale,
            "feature_stride_model_px": feature_stride, "effective_stride_original_px": eff}
    warns = []
    if scale < 1.0:
        warns.append(f"Tiles are DOWNSCALED x{scale:.2f} before the model: features below ~{1/scale:.1f}px may vanish.")
    warns.append(f"Model feature stride = {feature_stride}px at model input = {eff:.1f}px in the original image. "
                 f"Defects much smaller than this (e.g. 1-3px black spots) are likely to be missed by CLIP-type models.")
    warns.append("Zero-shot scores are uncalibrated similarity scores, NOT defect probabilities.")
    return warns, info


def map_candidates(alg, image: np.ndarray, score: np.ndarray, thr: float, class_name: str, max_count: int,
                   min_area: int = 1) -> list[Candidate]:
    out = []
    for c in ops.components(score, thr, min_area, 10**12, max_count):
        out.append(Candidate(None, f"{alg.algorithm_id}@{alg.version}", alg.algorithm_id, class_name, c["score"],
                             "bbox", {"x0": c["bbox"][0], "y0": c["bbox"][1], "x1": c["bbox"][2], "y1": c["bbox"][3]},
                             area_px=c["area"], features={"peak": (c["peak_x"], c["peak_y"])}))
    return out
