"""Tile datasets built from partial annotations (no full-image resizing; native resolution tiles)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.annotations import AnnotationStore
from ..core.image_io import load_mono
from ..core.label_targets import TargetPolicy, build_targets
from ..core.preprocessing import InputNormalization
from ..core.project import Project
from ..core import geometry as geo
from .augmentations import AugConfig, augment


class TargetCache:
    """Per-frame targets cached on disk; invalidated automatically when annotations/scopes/policy change."""

    def __init__(self, project: Project, policy: TargetPolicy):
        self.p = project
        self.store = AnnotationStore(project)
        self.policy = policy
        self.dir = project.root / "cache" / "targets"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, frame_id: int) -> str:
        rows = self.p.db.query("SELECT id, updated_at, deleted, confirmed FROM annotations WHERE frame_id=? ORDER BY id",
                               (frame_id,))
        scopes = self.p.db.query("SELECT class_name, ts FROM review_scopes WHERE frame_id=? ORDER BY class_name",
                                 (frame_id,))
        cls = self.p.db.query("SELECT name, kind, train_channel FROM classes ORDER BY name")
        blob = json.dumps([[tuple(r) for r in rows], [tuple(s) for s in scopes], [tuple(c) for c in cls],
                           self.policy.to_dict()], default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:16]

    def get(self, frame_id: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        f = self.dir / f"{frame_id}_{self._key(frame_id)}.npz"
        if f.exists():
            d = np.load(f)
            return d["target"].astype(np.float32), d["valid"].astype(bool)
        t, v, _ = build_targets(self.store, frame_id, shape, self.policy)
        for old in self.dir.glob(f"{frame_id}_*.npz"):
            old.unlink(missing_ok=True)
        np.savez_compressed(f, target=t.astype(np.float16), valid=v.astype(np.uint8))
        return t, v


class TileDataset(Dataset):
    """Samples native-resolution tiles. A fraction of tiles is centred (with jitter) on positive labels; others
    are drawn uniformly from frames, rejecting tiles with too few labelled (valid) pixels."""

    def __init__(self, project: Project, frame_ids: list[int], tile: int, policy: TargetPolicy,
                 norm: InputNormalization | None, samples_per_epoch: int, pos_fraction: float = 0.5,
                 min_valid_fraction: float = 0.02, aug: AugConfig | None = None, seed: int = 0,
                 deterministic: bool = False):
        self.p = project
        self.frames = [project.frame(f) for f in frame_ids]
        self.tile = tile
        self.policy = policy
        self.norm = norm
        self.n = samples_per_epoch
        self.pos_fraction = pos_fraction
        self.min_valid = min_valid_fraction
        self.aug = aug
        self.seed = seed
        self.epoch = 0
        self.deterministic = deterministic
        self.cache = TargetCache(project, policy)
        store = AnnotationStore(project)
        cmap = project.class_map()
        self.positives = []  # (frame list index, x, y)
        for i, f in enumerate(self.frames):
            for a in store.for_frame(f["id"], confirmed_only=True):
                if a["label_state"] == "positive" and cmap.get(a["class_name"], {}).get("train_channel") in policy.channels:
                    x, y = geo.geometry_center(a["geometry_type"], a["geometry"])
                    self.positives.append((i, x, y))
        self.spot_channel = (list(policy.channels).index("black_spot")
                             if "black_spot" in policy.channels and "black_spot" in policy.point_channels else None)

    def __len__(self):
        return self.n

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def _norm_for(self, f: dict) -> InputNormalization:
        if self.norm is not None:
            return self.norm
        return InputNormalization(bit_depth=f.get("bit_depth") or None)

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, self.epoch if not self.deterministic else 0, idx))
        for _attempt in range(20):
            if self.positives and rng.random() < self.pos_fraction:
                fi, px, py = self.positives[int(rng.integers(len(self.positives)))]
                f = self.frames[fi]
                jitter = self.tile // 3
                cx = px + rng.integers(-jitter, jitter + 1)
                cy = py + rng.integers(-jitter, jitter + 1)
            else:
                fi = int(rng.integers(len(self.frames)))
                f = self.frames[fi]
                cx = rng.uniform(0, f["width"]); cy = rng.uniform(0, f["height"])
            img = load_mono(f["path"])
            h, w = img.shape
            t = min(self.tile, h, w)
            x0 = int(np.clip(round(cx - t / 2), 0, w - t))
            y0 = int(np.clip(round(cy - t / 2), 0, h - t))
            target, valid = self.cache.get(f["id"], (h, w))
            tv = valid[:, y0:y0 + t, x0:x0 + t]
            if tv.mean() >= self.min_valid or _attempt == 19:
                break
        x = self._norm_for(f).apply(img[y0:y0 + t, x0:x0 + t])
        tt = target[:, y0:y0 + t, x0:x0 + t]
        n_syn = 0
        if self.aug is not None:
            x, tt, tv, n_syn = augment(x, tt, tv, self.aug, rng, self.spot_channel, self.policy.point_sigma)
        if t < self.tile:  # pad small images; padded area is invalid (unknown)
            ph, pw = self.tile - t, self.tile - t
            x = np.pad(x, ((0, ph), (0, pw)), mode="edge")
            tt = np.pad(tt, ((0, 0), (0, ph), (0, pw)))
            tv = np.pad(tv, ((0, 0), (0, ph), (0, pw)))
        return {"image": torch.from_numpy(np.ascontiguousarray(x))[None].float(),
                "target": torch.from_numpy(np.ascontiguousarray(tt)).float(),
                "valid": torch.from_numpy(np.ascontiguousarray(tv)).bool(),
                "frame_id": f["id"], "x0": x0, "y0": y0, "n_synthetic": n_syn}


# ------------------------------------------------------------------------------------------- patch dataset
def patch_samples(project: Project, frame_ids: list[int], class_names: list[str], negatives_per_frame: int = 8,
                  seed: int = 0) -> list[tuple[int, float, float, int]]:
    """(frame_id, x, y, class_index) samples from confirmed annotations + random points in exhaustively reviewed
    frames ('Verified Normal')."""
    store = AnnotationStore(project)
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(class_names)}
    out = []
    for fid in frame_ids:
        anns = store.for_frame(fid, confirmed_only=True)
        for a in anns:
            if a["label_state"] not in ("positive", "negative"):
                continue
            cname = a["class_name"]
            if a["label_state"] == "negative" and cname not in idx:
                cname = "Verified Normal"
            if cname not in idx:
                continue
            x, y = geo.geometry_center(a["geometry_type"], a["geometry"])
            out.append((fid, x, y, idx[cname]))
        if store.review_scopes(fid) and "Verified Normal" in idx:
            f = project.frame(fid)
            pos = [geo.geometry_center(a["geometry_type"], a["geometry"]) for a in anns
                   if a["label_state"] in ("positive", "unknown", "ignore")]
            k = 0
            while k < negatives_per_frame:
                x, y = rng.uniform(0, f["width"]), rng.uniform(0, f["height"])
                if all((x - px) ** 2 + (y - py) ** 2 > 32 ** 2 for px, py in pos):
                    out.append((fid, x, y, idx["Verified Normal"]))
                k += 1
    return out


def crop_multiscale(img: np.ndarray, x: float, y: float, crop_sizes, branch_px: int,
                    norm: InputNormalization) -> list[np.ndarray]:
    import cv2
    h, w = img.shape
    out = []
    for s in crop_sizes:
        half = s // 2
        cx, cy = int(x), int(y)
        x0, y0 = cx - half, cy - half
        pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
        src = np.pad(img, pad, mode="reflect") if pad else img
        c = src[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
        c = norm.apply(c)
        if s != branch_px:
            c = cv2.resize(c, (branch_px, branch_px), interpolation=cv2.INTER_AREA if s > branch_px else cv2.INTER_NEAREST)
        out.append(c.astype(np.float32))
    return out


class PatchDataset(Dataset):
    def __init__(self, project: Project, samples, crop_sizes, branch_px, norm: InputNormalization | None,
                 augment_: bool = True, seed: int = 0):
        self.p, self.samples = project, samples
        self.crop_sizes, self.branch_px, self.norm = crop_sizes, branch_px, norm
        self.augment, self.seed = augment_, seed
        self._frames = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fid, x, y, label = self.samples[i]
        f = self._frames.get(fid) or self._frames.setdefault(fid, self.p.frame(fid))
        img = load_mono(f["path"])
        rng = np.random.default_rng((self.seed, i))
        if self.augment:
            x += rng.integers(-1, 2); y += rng.integers(-1, 2)
        norm = self.norm or InputNormalization(bit_depth=f.get("bit_depth") or None)
        crops = crop_multiscale(img, x, y, self.crop_sizes, self.branch_px, norm)
        if self.augment:
            g = rng.uniform(0.93, 1.07)
            flip = rng.random() < 0.5
            crops = [(c[:, ::-1] if flip else c) * g for c in crops]
        return [torch.from_numpy(np.ascontiguousarray(c))[None] for c in crops], label
