"""Trainers (run in a separate process; the GUI only launches them and tails progress).

Progress protocol: each line ``@@PROGRESS {json}`` on stdout + the same records in <run_dir>/metrics.jsonl.
Graceful stop: create the file <run_dir>/STOP. Resume: pass resume=<run_dir>/last.pt.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..algorithms.supervised.models import HRUNet, MultiScalePatchCNN
from ..core.device import memory_report, select_device
from ..core.label_targets import TargetPolicy
from ..core.preprocessing import InputNormalization
from ..core.project import Project
from ..core.splits import split_bottles, split_id_by_name
from .augmentations import AugConfig
from .datasets import PatchDataset, TileDataset, patch_samples
from .losses import combined_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def emit(run_dir: Path, rec: dict) -> None:
    rec = {"t": time.time(), **rec}
    line = json.dumps(rec, default=float)
    print("@@PROGRESS " + line, flush=True)
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class UNetTrainConfig:
    project: str = ""
    split: str = ""
    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tile: int = 256
    samples_per_epoch: int = 512
    val_samples: int = 128
    pos_fraction: float = 0.5
    min_valid_fraction: float = 0.02
    base_channels: int = 16
    depth: int = 4
    context_module: bool = True
    loss: dict = field(default_factory=lambda: {"black_spot": "heatmap_focal", "bubble": "bce+dice",
                                                 "intensity": "bce+dice"})
    pos_weight: float | None = None
    policy: dict = field(default_factory=lambda: TargetPolicy().to_dict())
    normalization: dict = field(default_factory=lambda: InputNormalization().to_dict())
    use_session_bit_depth: bool = True
    augment: dict = field(default_factory=lambda: AugConfig().to_dict())
    device: str = "auto"
    amp: bool = False
    seed: int = 0
    num_workers: int = 0
    run_dir: str = ""
    resume: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "UNetTrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _frames_for(project: Project, split_id: int, subset: str) -> list[int]:
    return [f["id"] for f in project.all_frames(split_bottles(project, split_id, subset))]


def train_unet(cfg: UNetTrainConfig) -> Path:
    seed_everything(cfg.seed)
    project = Project(cfg.project)
    run_dir = Path(cfg.run_dir) if cfg.run_dir else None
    run_rec = None
    if run_dir is None:
        run_rec = project.create_run("train", "supervised.unet", params=asdict(cfg))
        run_dir = run_rec["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    sid = split_id_by_name(project, cfg.split)
    tr_frames, va_frames = _frames_for(project, sid, "train"), _frames_for(project, sid, "val")
    if not tr_frames:
        raise RuntimeError("Training subset is empty")
    policy = TargetPolicy.from_dict(cfg.policy)
    norm = None if cfg.use_session_bit_depth else InputNormalization.from_dict(cfg.normalization)
    aug = AugConfig.from_dict(cfg.augment)
    ds_tr = TileDataset(project, tr_frames, cfg.tile, policy, norm, cfg.samples_per_epoch, cfg.pos_fraction,
                        cfg.min_valid_fraction, aug, cfg.seed)
    ds_va = TileDataset(project, va_frames, cfg.tile, policy, norm, cfg.val_samples, cfg.pos_fraction,
                        cfg.min_valid_fraction, None, cfg.seed + 1, deterministic=True) if va_frames else None
    g = torch.Generator(); g.manual_seed(cfg.seed)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, generator=g)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, num_workers=cfg.num_workers) if ds_va else None

    device = select_device(cfg.device)
    channels = list(policy.channels)
    model = HRUNet(1, len(channels), cfg.base_channels, cfg.depth, cfg.context_module).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best = 0, float("inf")
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best_val", float("inf"))
        emit(run_dir, {"event": "resumed", "epoch": start_epoch})
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"))
    except Exception:  # tensorboard optional
        tb = None
    meta = {"arch": "HRUNet", "model_cfg": model.cfg, "channels": channels, "policy": policy.to_dict(),
            "normalization": None if norm is None else norm.to_dict(), "use_session_bit_depth": norm is None,
            "tile": cfg.tile, "train_config": asdict(cfg), "split": cfg.split,
            "score_semantics": "sigmoid output of U-Net trained with masked loss on partial labels (uncalibrated)"}
    emit(run_dir, {"event": "start", "device": str(device), "n_train_frames": len(tr_frames),
                   "n_val_frames": len(va_frames), "params": sum(p.numel() for p in model.parameters())})
    for epoch in range(start_epoch, cfg.epochs):
        if (run_dir / "STOP").exists():
            emit(run_dir, {"event": "stopped", "epoch": epoch}); break
        model.train(); ds_tr.set_epoch(epoch)
        tl, n, nsyn, t0 = 0.0, 0, 0, time.time()
        for step, b in enumerate(dl_tr):
            x, t, v = b["image"].to(device), b["target"].to(device), b["valid"].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
            loss, parts = combined_loss(out.float(), t, v, cfg.loss, channels, cfg.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += float(loss); n += 1; nsyn += int(b["n_synthetic"].sum())
            if (run_dir / "STOP").exists():
                break
            emit(run_dir, {"event": "step", "epoch": epoch, "step": step, "steps": len(dl_tr), "loss": float(loss)})
        sched.step()
        train_loss = tl / max(1, n)
        val_loss = None
        if dl_va is not None:
            model.eval(); vl, vn = 0.0, 0
            with torch.inference_mode():
                for b in dl_va:
                    out = model(b["image"].to(device))
                    l, _ = combined_loss(out.float(), b["target"].to(device), b["valid"].to(device), cfg.loss,
                                         channels, cfg.pos_weight)
                    vl += float(l); vn += 1
            val_loss = vl / max(1, vn)
        crit = val_loss if val_loss is not None else train_loss
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
              "epoch": epoch, "best_val": min(best, crit), "meta": meta}
        torch.save(ck, run_dir / "last.pt")
        is_best = crit < best
        if is_best:
            best = crit
            torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta, "criterion": crit},
                       run_dir / "best.pt")
        rec = {"event": "epoch", "epoch": epoch, "epochs": cfg.epochs, "train_loss": train_loss, "val_loss": val_loss,
               "best": is_best, "lr": sched.get_last_lr()[0], "sec": time.time() - t0, "synthetic_spots": nsyn,
               **memory_report(device)}
        emit(run_dir, rec)
        if tb:
            tb.add_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                tb.add_scalar("loss/val", val_loss, epoch)
    if tb:
        tb.close()
    emit(run_dir, {"event": "done", "best": best, "checkpoint": str(run_dir / "best.pt")})
    if run_rec:
        project.finish_run(run_rec["id"], "done")
    return run_dir / "best.pt"


# ------------------------------------------------------------------------------------------------ patch CNN
@dataclass
class PatchTrainConfig:
    project: str = ""
    split: str = ""
    classes: list = field(default_factory=lambda: ["Black Spot", "Bubble", "Intensity / Transparency Defect",
                                                    "Normal Embossing", "Normal Engraving", "Normal Mold Pattern",
                                                    "Dust", "Scratch", "Verified Normal"])
    crop_sizes: list = field(default_factory=lambda: [32, 128, 256])
    branch_px: int = 64
    width: int = 32
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    negatives_per_frame: int = 8
    device: str = "auto"
    seed: int = 0
    run_dir: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "PatchTrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def train_patch_cnn(cfg: PatchTrainConfig) -> Path:
    seed_everything(cfg.seed)
    project = Project(cfg.project)
    run_rec = None
    if cfg.run_dir:
        run_dir = Path(cfg.run_dir)
    else:
        run_rec = project.create_run("train", "supervised.patch_cnn", params=asdict(cfg))
        run_dir = run_rec["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    sid = split_id_by_name(project, cfg.split)
    s_tr = patch_samples(project, _frames_for(project, sid, "train"), cfg.classes, cfg.negatives_per_frame, cfg.seed)
    s_va = patch_samples(project, _frames_for(project, sid, "val"), cfg.classes, cfg.negatives_per_frame, cfg.seed + 1)
    if not s_tr:
        raise RuntimeError("No labelled patch samples in the training subset")
    counts = np.bincount([s[3] for s in s_tr], minlength=len(cfg.classes))
    emit(run_dir, {"event": "start", "n_train": len(s_tr), "n_val": len(s_va),
                   "class_counts": dict(zip(cfg.classes, counts.tolist()))})
    device = select_device(cfg.device)
    model = MultiScalePatchCNN(len(cfg.classes), cfg.crop_sizes, cfg.branch_px, cfg.width).to(device)
    w = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
    w = (w / w.sum() * len(cfg.classes)).to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    dl = DataLoader(PatchDataset(project, s_tr, cfg.crop_sizes, cfg.branch_px, None, True, cfg.seed),
                    batch_size=cfg.batch_size, shuffle=True)
    dlv = DataLoader(PatchDataset(project, s_va, cfg.crop_sizes, cfg.branch_px, None, False, cfg.seed),
                     batch_size=cfg.batch_size) if s_va else None
    meta = {"arch": "MultiScalePatchCNN", "model_cfg": model.cfg, "classes": cfg.classes,
            "train_config": asdict(cfg), "use_session_bit_depth": True}
    best = -float("inf")
    for epoch in range(cfg.epochs):
        if (run_dir / "STOP").exists():
            break
        model.train(); tl = 0.0
        for crops, y in dl:
            out = model([c.to(device) for c in crops])
            loss = lossf(out, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss)
        acc = None
        if dlv:
            model.eval(); ok = tot = 0
            with torch.inference_mode():
                for crops, y in dlv:
                    pr = model([c.to(device) for c in crops]).argmax(1).cpu()
                    ok += int((pr == y).sum()); tot += len(y)
            acc = ok / max(1, tot)
        crit = acc if acc is not None else -tl
        torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta}, run_dir / "last.pt")
        if crit > best:
            best = crit
            torch.save({"model": model.state_dict(), "epoch": epoch, "meta": meta}, run_dir / "best.pt")
        emit(run_dir, {"event": "epoch", "epoch": epoch, "epochs": cfg.epochs, "train_loss": tl / max(1, len(dl)),
                       "val_acc": acc})
    emit(run_dir, {"event": "done", "checkpoint": str(run_dir / "best.pt")})
    if run_rec:
        project.finish_run(run_rec["id"], "done")
    return run_dir / "best.pt"
