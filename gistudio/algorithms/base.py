"""Common algorithm interface, standard result objects and the algorithm registry.

Every inspection method (classical, zero-shot, supervised, anomaly) implements `InspectionAlgorithm.run`
and returns an `InspectionResult`. Score semantics differ per model: `score_semantics` documents what the
score means; scores of different models are NOT comparable probabilities.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParamSpec:
    name: str
    type: str                      # int | float | bool | choice | str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""


@dataclass
class Candidate:
    frame_id: int | None
    model_id: str
    source_algorithm: str
    class_name: str
    score: float
    geometry_type: str             # point | bbox | mask(bbox-only summary stored)
    geometry: dict
    area_px: float = 0.0
    features: dict = field(default_factory=dict)
    review_status: str = "pending"
    candidate_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InspectionResult:
    candidates: list[Candidate]
    score_maps: dict[str, np.ndarray] = field(default_factory=dict)   # name -> float32 (H, W), original coords
    inference_ms: float = 0.0
    preprocess: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class InspectionAlgorithm:
    algorithm_id: str = "base"
    display_name: str = "Base"
    family: str = "classical"      # classical | zero_shot | supervised | anomaly
    version: str = "1"
    default_class: str = "Black Spot"
    score_semantics: str = ""
    params_spec: list[ParamSpec] = []

    def __init__(self, **params):
        self.params = {p.name: p.default for p in self.params_spec}
        unknown = set(params) - set(self.params)
        if unknown:
            raise KeyError(f"{self.algorithm_id}: unknown params {unknown}")
        self.params.update(params)

    def run(self, image: np.ndarray, meta: dict | None = None) -> InspectionResult:
        t0 = time.perf_counter()
        res = self._run(image, meta or {})
        res.inference_ms = (time.perf_counter() - t0) * 1000.0
        res.params = dict(self.params)
        for c in res.candidates:
            c.features.setdefault("inference_ms_frame", res.inference_ms)
        return res

    def _run(self, image: np.ndarray, meta: dict) -> InspectionResult:  # pragma: no cover
        raise NotImplementedError

    def info(self) -> dict:
        return {"algorithm_id": self.algorithm_id, "family": self.family, "version": self.version,
                "score_semantics": self.score_semantics, "params": dict(self.params)}


_REGISTRY: dict[str, type[InspectionAlgorithm]] = {}


def register(cls: type[InspectionAlgorithm]) -> type[InspectionAlgorithm]:
    if cls.algorithm_id in _REGISTRY and _REGISTRY[cls.algorithm_id] is not cls:
        raise KeyError(f"duplicate algorithm id {cls.algorithm_id}")
    _REGISTRY[cls.algorithm_id] = cls
    return cls


def registry(family: str | None = None) -> dict[str, type[InspectionAlgorithm]]:
    _load_builtin()
    return {k: v for k, v in _REGISTRY.items() if family is None or v.family == family}


def create(algorithm_id: str, **params) -> InspectionAlgorithm:
    _load_builtin()
    return _REGISTRY[algorithm_id](**params)


_loaded = False


def _load_builtin() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from .classical import algorithms as _c  # noqa: F401
    from .zero_shot import plugins as _z  # noqa: F401
    from .supervised import plugins as _s  # noqa: F401
    _load_entry_plugins()


def _load_entry_plugins() -> None:
    """Third-party algorithms can register via the 'gistudio.algorithms' entry-point group or by dropping a
    module into <repo>/plugins/ that calls `register`."""
    import importlib
    import importlib.metadata as md
    import sys
    from pathlib import Path
    try:
        for ep in md.entry_points(group="gistudio.algorithms"):
            ep.load()
    except Exception:
        pass
    pdir = Path(__file__).resolve().parents[2] / "plugins"
    if pdir.is_dir():
        sys.path.insert(0, str(pdir))
        for f in sorted(pdir.glob("*.py")):
            if not f.name.startswith("_"):
                try:
                    importlib.import_module(f.stem)
                except Exception as e:  # plugins must not break the app
                    print(f"[plugins] failed to load {f.name}: {e!r}")


Progress = Callable[[int, int, str], None]
