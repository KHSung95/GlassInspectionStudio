"""Supervised model inference wrapped in the common algorithm interface."""
from __future__ import annotations

import numpy as np

from ..base import Candidate, InspectionAlgorithm, InspectionResult, ParamSpec, create, register
from ..classical import ops

CHANNEL_CLASS = {"black_spot": "Black Spot", "bubble": "Bubble", "intensity": "Intensity / Transparency Defect",
                 "linear_void": "Linear / Broken Void", "tint": "Faint Tint"}
_CK_CACHE: dict = {}


def load_checkpoint(path: str, device):
    import torch
    key = (str(path), str(device))
    if key not in _CK_CACHE:
        ck = torch.load(path, map_location=device, weights_only=False)
        meta = ck["meta"]
        if meta["arch"] == "HRUNet":
            from .models import HRUNet
            m = HRUNet(**meta["model_cfg"])
        elif meta["arch"] == "MultiScalePatchCNN":
            from .models import MultiScalePatchCNN
            m = MultiScalePatchCNN(**meta["model_cfg"])
        else:
            raise ValueError(meta["arch"])
        m.load_state_dict(ck["model"])
        _CK_CACHE[key] = (m.eval().to(device), meta)
    return _CK_CACHE[key]


def peak_candidates(score: np.ndarray, thr: float, nms_radius: int, max_count: int) -> list[tuple[float, float, float]]:
    """Local maxima above thr with non-maximum suppression (radius in px)."""
    import cv2
    k = 2 * nms_radius + 1
    dil = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ys, xs = np.nonzero((score >= thr) & (score >= dil))
    vals = score[ys, xs]
    order = np.argsort(-vals)[:max_count]
    return [(xs[i] + 0.5, ys[i] + 0.5, float(vals[i])) for i in order]


@register
class UNetInference(InspectionAlgorithm):
    algorithm_id = "supervised.unet"
    display_name = "High-resolution U-Net (checkpoint)"
    family = "supervised"
    score_semantics = "U-Net sigmoid output (uncalibrated; trained with masked loss on partial labels)"
    params_spec = [
        ParamSpec("checkpoint", "str", "", help="Path to best.pt"),
        ParamSpec("tile", "int", 512, 64, 4096, 32),
        ParamSpec("overlap", "int", 64, 0, 1024, 8),
        ParamSpec("batch", "int", 4, 1, 64),
        ParamSpec("threshold_spot", "float", 0.3, 0.0, 1.0, 0.01),
        ParamSpec("threshold_area", "float", 0.5, 0.0, 1.0, 0.01),
        ParamSpec("nms_radius", "int", 4, 1, 64),
        ParamSpec("min_area", "int", 4, 1, 10**6),
        ParamSpec("max_candidates", "int", 500, 1, 100000),
        ParamSpec("device", "str", "auto"),
    ]

    def _run(self, image, meta):
        import torch
        from ...core.device import select_device
        from ...core.preprocessing import InputNormalization
        from ...core.tiling import Stitcher, make_tiles, pad_to
        p = self.params
        if not p["checkpoint"]:
            raise ValueError("checkpoint path is required")
        dev = select_device(p["device"])
        model, cm = load_checkpoint(p["checkpoint"], dev)
        norm = (InputNormalization(bit_depth=meta.get("bit_depth")) if cm.get("use_session_bit_depth", True)
                else InputNormalization.from_dict(cm["normalization"]))
        x = norm.apply(image)
        h, w = x.shape
        tiles = make_tiles(h, w, p["tile"], p["overlap"])
        st = Stitcher(h, w, len(cm["channels"]), mode="mean", window=True)
        for i in range(0, len(tiles), p["batch"]):
            chunk = tiles[i:i + p["batch"]]
            arrs = [pad_to(x[t.y0:t.y1, t.x0:t.x1], max(t.w, t.h))[0] for t in chunk]
            size = max(a.shape[0] for a in arrs)
            arrs = [pad_to(a, size)[0] for a in arrs]
            with torch.inference_mode():
                out = torch.sigmoid(model(torch.from_numpy(np.stack(arrs))[:, None].to(dev))).cpu().numpy()
            for t, o in zip(chunk, out):
                st.add(t, o[:, :t.h, :t.w])
            if meta.get("progress"):
                meta["progress"](min(i + p["batch"], len(tiles)), len(tiles), "tiles")
        maps = st.result()
        cands = []
        point_ch = set(cm["policy"].get("point_channels", ["black_spot"]))
        mid = f"{self.algorithm_id}@{cm['train_config'].get('run_dir', '') or p['checkpoint']}"
        for ci, ch in enumerate(cm["channels"]):
            cls = CHANNEL_CLASS.get(ch, ch)
            if ch in point_ch:
                for (cx, cy, s) in peak_candidates(maps[ci], p["threshold_spot"], p["nms_radius"], p["max_candidates"]):
                    cands.append(Candidate(None, mid, self.algorithm_id, cls, s, "point", {"x": cx, "y": cy, "r": 2.0},
                                           area_px=1.0, features={"channel": ch}))
            else:
                for c in ops.components(maps[ci], p["threshold_area"], p["min_area"], 10**12, p["max_candidates"]):
                    cands.append(Candidate(None, mid, self.algorithm_id, cls, c["score"], "bbox",
                                           {"x0": c["bbox"][0], "y0": c["bbox"][1], "x1": c["bbox"][2], "y1": c["bbox"][3]},
                                           area_px=c["area"], features={"channel": ch, "peak": (c["peak_x"], c["peak_y"])}))
        return InspectionResult(cands, {f"unet_{ch}": maps[i] for i, ch in enumerate(cm["channels"])},
                                preprocess={"normalization": norm.to_dict()},
                                model_info={"checkpoint": p["checkpoint"], "channels": cm["channels"],
                                            "tile": p["tile"], "overlap": p["overlap"], "train_split": cm.get("split")})


@register
class PatchCNNCascade(InspectionAlgorithm):
    algorithm_id = "supervised.patch_cnn"
    display_name = "Classical front-end + Multi-scale Patch CNN"
    family = "supervised"
    score_semantics = "Patch CNN softmax mass on defect classes (uncalibrated); front-end misses are unrecoverable"
    params_spec = [
        ParamSpec("checkpoint", "str", ""),
        ParamSpec("frontend", "str", "classical.log_blob"),
        ParamSpec("frontend_threshold", "float", 4.0, 0.0, 100, 0.5),
        ParamSpec("threshold", "float", 0.5, 0.0, 1.0, 0.01),
        ParamSpec("device", "str", "auto"),
    ]

    def _run(self, image, meta):
        import torch
        from ...core.device import select_device
        from ...core.preprocessing import InputNormalization
        from ...training.datasets import crop_multiscale
        p = self.params
        dev = select_device(p["device"])
        model, cm = load_checkpoint(p["checkpoint"], dev)
        fe = create(p["frontend"], threshold=p["frontend_threshold"])
        fres = fe.run(image, meta)
        norm = InputNormalization(bit_depth=meta.get("bit_depth"))
        classes = cm["classes"]
        defect_idx = [i for i, c in enumerate(classes) if c in ("Black Spot", "Bubble", "Intensity / Transparency Defect",
                                                                 "Open Bubble", "Closed Bubble", "Other Defect")]
        out = []
        if fres.candidates:
            from ...core import geometry as geo
            batch = []
            for c in fres.candidates:
                x, y = geo.geometry_center(c.geometry_type, c.geometry)
                batch.append(crop_multiscale(image, x, y, cm["model_cfg"]["crop_sizes"], cm["model_cfg"]["branch_px"], norm))
            with torch.inference_mode():
                crops = [torch.from_numpy(np.stack([b[k] for b in batch]))[:, None].to(dev)
                         for k in range(len(cm["model_cfg"]["crop_sizes"]))]
                prob = torch.softmax(model(crops), 1).cpu().numpy()
            for c, pr in zip(fres.candidates, prob):
                s = float(pr[defect_idx].sum())
                top = classes[int(pr.argmax())]
                if s >= p["threshold"]:
                    out.append(Candidate(None, f"{self.algorithm_id}@{p['checkpoint']}", self.algorithm_id,
                                         top if top in [classes[i] for i in defect_idx] else c.class_name, s,
                                         c.geometry_type, c.geometry, c.area_px,
                                         {"frontend_score": c.score, "class_probs": dict(zip(classes, pr.round(4).tolist()))}))
        return InspectionResult(out, {}, model_info={"checkpoint": p["checkpoint"], "frontend": fe.info(),
                                                     "frontend_candidates": len(fres.candidates)},
                                warnings=["Cascade recall is bounded by the front-end detector recall; evaluate the "
                                          "front-end run separately."])
