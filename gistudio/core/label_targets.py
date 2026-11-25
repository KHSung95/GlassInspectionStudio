"""Convert partial annotations into supervised targets + per-class validity masks.

For each output channel c we build:
  target[c] : float32 in [0, 1]  (heatmap for point classes, binary mask otherwise)
  valid[c]  : bool — pixels where the label for channel c is KNOWN. Loss is computed only here.

Rules (applied in this order, later rules override earlier ones):
  1. Default: valid = False everywhere (UNKNOWN). Absence of a positive never implies normal.
  2. Review scope (class exhaustively reviewed on this frame) -> valid[c] = True everywhere, target 0.
  3. Negative annotations (any normal / verified-normal class, or appearance class with verdict 'ok')
     -> valid = True, target = 0 for ALL defect channels in that region.
  4. Positive annotations of a defect class mapped to channel c -> target = 1 (or Gaussian), valid[c] = True
     in the positive support. For point labels the valid support is a disc of radius `point_valid_radius`
     around the point (the immediate surroundings are known to be "the defect and its rim").
  5. Unknown annotations, unconfirmed annotations, appearance classes with undecided verdict -> valid = False.
  6. Ignore ROIs -> valid = False for all channels (highest priority).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import geometry as geo
from .annotations import AnnotationStore
from .project import TRAIN_CHANNELS


@dataclass
class TargetPolicy:
    channels: tuple[str, ...] = tuple(TRAIN_CHANNELS)
    point_sigma: float = 1.5                 # Gaussian sigma (px) of the training heatmap — NOT defect size
    point_valid_radius: float = 6.0          # disc around a point label considered labelled
    point_channels: tuple[str, ...] = ("black_spot",)   # channels trained as heatmaps
    point_as_disc_radius: float = 2.0        # for non-heatmap channels, a point becomes a disc of this radius

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetPolicy":
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


def build_targets(store: AnnotationStore, frame_id: int, shape: tuple[int, int],
                  policy: TargetPolicy | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    policy = policy or TargetPolicy()
    h, w = shape
    C = len(policy.channels)
    target = np.zeros((C, h, w), np.float32)
    valid = np.zeros((C, h, w), bool)
    cmap = store.p.class_map()
    ch_index = {c: i for i, c in enumerate(policy.channels)}
    anns = store.for_frame(frame_id, confirmed_only=True)
    info = {"n_pos": 0, "n_neg": 0, "n_ignore": 0, "n_unknown": 0, "scopes": []}

    # 2. review scopes
    for cls_name in store.review_scopes(frame_id):
        ch = cmap.get(cls_name, {}).get("train_channel", "")
        if ch in ch_index:
            valid[ch_index[ch]] = True
            info["scopes"].append(cls_name)

    def is_negative(a) -> bool:
        k = cmap.get(a["class_name"], {}).get("kind")
        if a["label_state"] == "negative":
            return True
        # observed appearance (dust/scratch) explicitly judged acceptable by spec -> negative for defect channels
        return a["label_state"] == "positive" and k == "appearance" and a.get("verdict") == "ok"

    # 3. negatives
    for a in anns:
        if is_negative(a):
            m = store.rasterize(a, shape)
            valid[:, m] = True
            target[:, m] = 0.0
            info["n_neg"] += 1

    # 4. positives
    point_pts: dict[int, list] = {}
    for a in anns:
        if a["label_state"] != "positive":
            continue
        cls = cmap.get(a["class_name"], {})
        ch = cls.get("train_channel", "")
        if cls.get("kind") != "defect" or ch not in ch_index:
            continue
        ci = ch_index[ch]
        info["n_pos"] += 1
        if a["geometry_type"] == "point":
            if ch in policy.point_channels:
                point_pts.setdefault(ci, []).append((a["geometry"]["x"], a["geometry"]["y"]))
                support = geo.rasterize("point", a["geometry"], shape, point_radius=policy.point_valid_radius)
                valid[ci, support] = True
                target[ci, support] = 0.0  # heatmap written below
            else:
                disc = geo.rasterize("point", a["geometry"], shape, point_radius=policy.point_as_disc_radius)
                valid[ci, disc] = True
                target[ci, disc] = 1.0
        else:
            m = store.rasterize(a, shape)
            valid[ci, m] = True
            target[ci, m] = 1.0
    for ci, pts in point_pts.items():
        hm = geo.gaussian_heatmap(shape, pts, policy.point_sigma)
        target[ci] = np.maximum(target[ci], hm)

    # 5. unknown / undecided appearance
    for a in anns:
        k = cmap.get(a["class_name"], {}).get("kind")
        if a["label_state"] == "unknown" or (k == "appearance" and a["label_state"] == "positive"
                                             and a.get("verdict") != "ok"):
            m = store.rasterize(a, shape)
            # positives already written stay valid only for their own channel; undecided appearance hides
            # defect channels in its region
            for ci in range(C):
                pos_here = target[ci] > 0
                valid[ci, m & ~pos_here] = False
            info["n_unknown"] += 1

    # 6. ignore
    for a in anns:
        if a["label_state"] == "ignore":
            m = store.rasterize(a, shape)
            valid[:, m] = False
            target[:, m] = 0.0
            info["n_ignore"] += 1
    return target, valid, info
