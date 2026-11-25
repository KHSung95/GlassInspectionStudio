"""Zero-shot inference plugins.

Implemented & verified:
  * zeroshot.winclip      — Anomalib 2.x `WinClipModel` (official implementation), zero-shot (k_shot=0) mode,
                            backbone ViT-B-16-plus-240 / laion400m_e31 from open_clip.
  * zeroshot.clip_patch   — OpenCLIP patch classification against user-editable text prompts.

Not integrated (see docs/MODELS.md for the verification notes):
  * AnomalyCLIP — no pip-installable official package; checkpoints are distributed via the authors' repository
                  only. Registered as `zeroshot.anomalyclip` placeholder that reports unavailability instead of
                  faking results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..base import InspectionAlgorithm, InspectionResult, ParamSpec, register
from .common import configure_model_cache, map_candidates, resolution_warnings, tiled_map

DEFAULT_PROMPTS = {
    "normal": [
        "a clean transparent glass surface",
        "a normal embossed marking on a glass bottle",
        "a normal engraved number on glass",
    ],
    "defect": [
        "a tiny black particle in transparent glass",
        "a dark inclusion inside glass",
        "a subtle cloudy region in transparent glass",
        "an air bubble trapped inside glass",
    ],
}

_MODEL_CACHE: dict = {}


def _device(pref: str):
    from ...core.device import select_device
    return select_device(pref)


@register
class WinCLIPZeroShot(InspectionAlgorithm):
    algorithm_id = "zeroshot.winclip"
    display_name = "WinCLIP (Anomalib, zero-shot)"
    family = "zero_shot"
    version = "anomalib-2.x"
    default_class = "Other Defect"
    score_semantics = "WinCLIP zero-shot anomaly score (softmax over normal/anomalous prompt ensembles); uncalibrated"
    params_spec = [
        ParamSpec("class_name", "str", "glass bottle", help="Object name used in WinCLIP's prompt ensemble"),
        ParamSpec("tile", "int", 240, 64, 4096, 16, help="Native-resolution tile (px) fed to the 240px model"),
        ParamSpec("overlap", "int", 48, 0, 2048, 8),
        ParamSpec("batch", "int", 8, 1, 128),
        ParamSpec("contrast", "choice", "fixed", choices=["fixed", "percentile"]),
        ParamSpec("threshold", "float", 0.6, 0.0, 1.0, 0.01),
        ParamSpec("max_candidates", "int", 100, 1, 100000),
        ParamSpec("device", "str", "auto"),
    ]
    MODEL_SIZE = 240
    FEATURE_STRIDE = 16

    def _model(self):
        key = ("winclip", self.params["class_name"], self.params["device"])
        if key not in _MODEL_CACHE:
            configure_model_cache()
            from anomalib.models.image.winclip.torch_model import WinClipModel
            m = WinClipModel(class_name=self.params["class_name"]).eval().to(_device(self.params["device"]))
            _MODEL_CACHE[key] = m
        return _MODEL_CACHE[key]

    def _run(self, image, meta):
        import torch
        p = self.params
        model = self._model()
        dev = next(model.parameters()).device

        def fn(x):
            with torch.inference_mode():
                out = model(x.to(dev))
            return out.anomaly_map.float().cpu().numpy()

        score = tiled_map(image, p["tile"], p["overlap"], self.MODEL_SIZE, p["batch"], fn, meta.get("bit_depth"),
                          p["contrast"], meta.get("progress"))
        warns, info = resolution_warnings(p["tile"], self.MODEL_SIZE, self.FEATURE_STRIDE)
        info.update({"backbone": "ViT-B-16-plus-240", "pretrained": "laion400m_e31", "implementation":
                     "anomalib.models.image.winclip.torch_model.WinClipModel", "k_shot": 0})
        cands = map_candidates(self, image, score, p["threshold"], self.default_class, p["max_candidates"])
        return InspectionResult(cands, {"winclip_score": score}, preprocess={"contrast": p["contrast"]},
                                model_info=info, warnings=warns)


@register
class CLIPPatchClassifier(InspectionAlgorithm):
    algorithm_id = "zeroshot.clip_patch"
    display_name = "OpenCLIP patch-prompt classification"
    family = "zero_shot"
    version = "openclip"
    default_class = "Other Defect"
    score_semantics = "sum of softmax probabilities of 'defect' prompts among all prompts (per tile); uncalibrated"
    params_spec = [
        ParamSpec("model", "str", "ViT-B-16-plus-240"),
        ParamSpec("pretrained", "str", "laion400m_e31"),
        ParamSpec("prompts_json", "str", "", help="Path to a prompt set JSON {normal:[...], defect:[...]}"),
        ParamSpec("tile", "int", 128, 32, 4096, 16),
        ParamSpec("overlap", "int", 64, 0, 2048, 8),
        ParamSpec("batch", "int", 16, 1, 256),
        ParamSpec("contrast", "choice", "fixed", choices=["fixed", "percentile"]),
        ParamSpec("temperature", "float", 100.0, 1.0, 1000.0, 1.0),
        ParamSpec("threshold", "float", 0.6, 0.0, 1.0, 0.01),
        ParamSpec("max_candidates", "int", 100, 1, 100000),
        ParamSpec("device", "str", "auto"),
    ]

    def prompts(self) -> dict:
        pj = self.params["prompts_json"]
        if pj and Path(pj).exists():
            return json.loads(Path(pj).read_text(encoding="utf-8"))
        return DEFAULT_PROMPTS

    def _model(self):
        key = ("clip", self.params["model"], self.params["pretrained"], self.params["device"])
        if key not in _MODEL_CACHE:
            configure_model_cache()
            import open_clip
            m, _, _ = open_clip.create_model_and_transforms(self.params["model"], pretrained=self.params["pretrained"])
            m = m.eval().to(_device(self.params["device"]))
            tok = open_clip.get_tokenizer(self.params["model"])
            _MODEL_CACHE[key] = (m, tok)
        return _MODEL_CACHE[key]

    def _run(self, image, meta):
        import torch
        p = self.params
        model, tok = self._model()
        dev = next(model.parameters()).device
        pr = self.prompts()
        texts = list(pr["normal"]) + list(pr["defect"])
        n_norm = len(pr["normal"])
        with torch.inference_mode():
            te = model.encode_text(tok(texts).to(dev)).float()
            te = te / te.norm(dim=-1, keepdim=True)
        size = model.visual.image_size
        size = size[0] if isinstance(size, (tuple, list)) else int(size)
        per_prompt: list[np.ndarray] = []

        def fn(x):
            with torch.inference_mode():
                ie = model.encode_image(x.to(dev)).float()
                ie = ie / ie.norm(dim=-1, keepdim=True)
                prob = (p["temperature"] * ie @ te.T).softmax(-1).cpu().numpy()   # (B, P)
            per_prompt.append(prob)
            s = prob[:, n_norm:].sum(1)
            return np.repeat(np.repeat(s[:, None, None], 2, 1), 2, 2)  # constant map per tile

        score = tiled_map(image, p["tile"], p["overlap"], size, p["batch"], fn, meta.get("bit_depth"),
                          p["contrast"], meta.get("progress"))
        probs = np.concatenate(per_prompt, 0) if per_prompt else np.zeros((0, len(texts)))
        warns, info = resolution_warnings(p["tile"], size, p["tile"])  # one embedding per tile
        warns[1] = (f"One embedding per tile: localisation granularity = tile stride ({p['tile'] - p['overlap']}px). "
                    "Tiny spots occupy a negligible fraction of the tile and are likely invisible to CLIP.")
        info.update({"model": p["model"], "pretrained": p["pretrained"], "prompts": pr,
                     "mean_prompt_prob": dict(zip(texts, probs.mean(0).round(4).tolist())) if len(probs) else {}})
        cands = map_candidates(self, image, score, p["threshold"], self.default_class, p["max_candidates"])
        return InspectionResult(cands, {"clip_defect_prob": score}, preprocess={"contrast": p["contrast"]},
                                model_info=info, warnings=warns)


@register
class AnomalyCLIPPlaceholder(InspectionAlgorithm):
    algorithm_id = "zeroshot.anomalyclip"
    display_name = "AnomalyCLIP (not available)"
    family = "zero_shot"
    version = "unavailable"
    available = False
    score_semantics = "n/a"
    params_spec = []

    def _run(self, image, meta):
        raise RuntimeError(
            "AnomalyCLIP is not integrated: no official pip package / verified checkpoint was installed. "
            "See docs/MODELS.md for the verification notes and how to add it as a plugin.")
