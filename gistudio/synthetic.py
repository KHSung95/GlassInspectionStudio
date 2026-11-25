"""Synthetic rotating transparent-bottle sequences for SOFTWARE VERIFICATION ONLY.

This does not model real optics faithfully and says nothing about real detection performance.

Model: a vertical cylinder of radius R (px) rendered from a texture in cylinder coordinates (phi, y).
  * front surface sampled at phi = asin((x-cx)/R) - theta, back surface (attenuated, mirrored) superimposed
    -> front/back overlap, foreshortening near the silhouette, defects moving with rotation;
  * normal structures: graduation ticks + embossed digits (dark line + bright 1px highlight), shoulder/bottom
    refraction bands, silhouette edge darkening;
  * defects: 1-5px black spots, low-contrast blotches, bubbles (bright core + dark rim), defects on top of
    normal structures, defects with angle-dependent visibility windows;
  * 16-bit containers with 12-bit (default) or full 16-bit data, Gaussian + shot noise.

Writes per session: frames (TIFF), angles.csv, session_meta.json, and ground_truth.json (ALL defects per
frame, including the ones deliberately left unlabelled) and labels.json (partial 'customer' labels).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile


@dataclass
class SynthDefect:
    kind: str              # black_spot | blotch | bubble
    phi: float             # azimuth (rad) on the cylinder
    y: float               # image row
    size: float            # radius (px)
    amplitude: float       # fractional intensity change at centre (negative = darker)
    labeled: bool = True   # included in partial labels?
    vis_center: float | None = None   # optional angular visibility window centre (deg of theta)
    vis_width: float | None = None    # window full width (deg)
    on_structure: bool = False
    uid: str = ""


@dataclass
class BottleSpec:
    code: str
    product_type: str
    radius: int
    height: int = 512
    width: int = 640
    bit_depth: int = 12
    digits: str = "10 20 30"
    defects: list[SynthDefect] = field(default_factory=list)


def _texture(spec: BottleSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (normal_texture, defect_texture_stack_placeholder). Texture value = multiplicative modulation-1."""
    n_phi = int(round(2 * math.pi * spec.radius))
    H = spec.height
    tex = np.zeros((H, n_phi), np.float32)
    # graduation ticks in a 50deg sector centred at phi=0, embossed digits beside them
    c0 = n_phi // 2  # phi=0 at texture centre (we index phi+pi)
    y0, y1 = int(H * 0.25), int(H * 0.75)
    for k, yy in enumerate(range(y0, y1, max(8, H // 24))):
        L = 30 if k % 5 == 0 else 16
        tex[yy, c0 - L:c0 + L] -= 0.22
        tex[yy + 1, c0 - L:c0 + L] += 0.10   # bright highlight (embossing)
    canvas = np.zeros((H, n_phi), np.uint8)
    for i, word in enumerate(spec.digits.split()):
        cv2.putText(canvas, word, (c0 + 40, y0 + 20 + i * ((y1 - y0) // 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1,
                    cv2.LINE_AA)
    d = canvas.astype(np.float32) / 255.0
    tex -= 0.18 * d
    tex[1:, :] += 0.07 * d[:-1, :]  # emboss highlight
    # mold seam line (vertical) at phi = +100deg
    seam = int((math.radians(100) + math.pi) / (2 * math.pi) * n_phi) % n_phi
    tex[:, seam] -= 0.05
    return tex, np.zeros_like(tex)


def _defect_layer(spec: BottleSpec, theta_deg: float) -> np.ndarray:
    n_phi = int(round(2 * math.pi * spec.radius))
    H = spec.height
    lay = np.zeros((H, n_phi), np.float32)
    for d in spec.defects:
        amp = d.amplitude
        if d.vis_center is not None:
            dd = abs(((theta_deg - d.vis_center) + 180) % 360 - 180)
            amp *= max(0.0, 1.0 - dd / (d.vis_width / 2)) if d.vis_width else 1.0
        if amp == 0:
            continue
        cx = ((d.phi + math.pi) % (2 * math.pi)) / (2 * math.pi) * n_phi
        cy = d.y
        R = int(4 * d.size) + 3
        xs = np.arange(int(cx) - R, int(cx) + R + 1)
        ys = np.arange(max(0, int(cy) - R), min(H, int(cy) + R + 1))
        X, Y = np.meshgrid(xs, ys)
        r2 = (X + 0.5 - cx) ** 2 + (Y + 0.5 - cy) ** 2
        if d.kind == "black_spot":
            v = amp * np.exp(-r2 / (2 * max(d.size, 0.4) ** 2))
        elif d.kind == "blotch":
            v = amp * np.exp(-r2 / (2 * d.size ** 2))
        elif d.kind == "bubble":
            r = np.sqrt(r2)
            v = (-amp * 0.6) * np.exp(-r2 / (2 * (0.5 * d.size) ** 2)) + amp * np.exp(-((r - d.size) ** 2) / 2.0)
        else:
            continue
        lay[np.ix_(ys, xs % n_phi)] += v.astype(np.float32)
    return lay


def render(spec: BottleSpec, theta_deg: float, rng: np.random.Generator, tex: np.ndarray) -> np.ndarray:
    H, W, R = spec.height, spec.width, spec.radius
    n_phi = tex.shape[1]
    full = (1 << spec.bit_depth) - 1
    layer = tex + _defect_layer(spec, theta_deg)
    cx = W / 2.0
    xs = np.arange(W) + 0.5
    u = np.clip((xs - cx) / R, -1, 1)
    inside = np.abs(xs - cx) < R
    th = math.radians(theta_deg)
    phi_f = np.arcsin(u) - th
    phi_b = math.pi - np.arcsin(u) - th
    col_f = (((phi_f + math.pi) % (2 * math.pi)) / (2 * math.pi) * n_phi - 0.5).astype(np.float32)
    col_b = (((phi_b + math.pi) % (2 * math.pi)) / (2 * math.pi) * n_phi - 0.5).astype(np.float32)
    rows = np.arange(H, dtype=np.float32)
    mxf = np.tile(col_f, (H, 1)); mxb = np.tile(col_b, (H, 1))
    my = np.tile(rows[:, None], (1, W))
    front = cv2.remap(layer, mxf, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    back = cv2.remap(layer, mxb, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    mod = np.where(inside[None, :], front + 0.35 * back, 0.0)
    base = np.full((H, W), 0.82, np.float32)  # backlight
    rr = np.abs(u)
    body = 0.74 * (1 - 0.55 * rr ** 8)
    base[:, inside] = body[inside]
    sh0, sh1 = int(H * 0.08), int(H * 0.14)
    base[sh0:sh1, inside] *= 0.65  # shoulder refraction band
    base[int(H * 0.93):, inside] *= 0.6  # bottom
    img = base * (1 + mod)
    sig = img * full
    noise = rng.normal(0, 1, sig.shape) * np.sqrt(np.maximum(sig, 0) * 0.02 + (0.002 * full) ** 2)
    out = np.clip(np.rint(sig + noise), 0, full).astype(np.uint16)
    return out


def project_defect(spec: BottleSpec, d: SynthDefect, theta_deg: float) -> dict | None:
    """Image location & visibility of a defect at rotation theta (front surface only for GT)."""
    a = d.phi + math.radians(theta_deg)
    c = math.cos(a)
    if c <= 0.15:  # back side / grazing: not reliably visible -> not in visible GT
        return None
    amp = d.amplitude
    if d.vis_center is not None and d.vis_width:
        dd = abs(((theta_deg - d.vis_center) + 180) % 360 - 180)
        amp *= max(0.0, 1.0 - dd / (d.vis_width / 2))
    if abs(amp) < 0.02:
        return None
    x = spec.width / 2.0 + spec.radius * math.sin(a)
    return {"x": x, "y": d.y + 0.5, "foreshortening": c, "effective_amplitude": amp}


def default_bottles(n: int = 4, width: int = 640, height: int = 512, seed: int = 0) -> list[BottleSpec]:
    rng = np.random.default_rng(seed)
    specs = []
    for b in range(n):
        ptype = "VIAL-10" if b % 2 == 0 else "VIAL-20"
        R = int(width * (0.30 if ptype == "VIAL-10" else 0.38))
        bit = 16 if b == n - 1 else 12
        s = BottleSpec(f"B{b + 1:03d}", ptype, R, height, width, bit, "5 10 15" if ptype == "VIAL-10" else "10 20 30")
        defs = []
        for i in range(4):  # black spots 1..5 px of varying contrast
            defs.append(SynthDefect("black_spot", float(rng.uniform(-math.pi, math.pi)),
                                    float(rng.uniform(height * 0.2, height * 0.85)), float(rng.choice([0.5, 0.8, 1.2, 2.0])),
                                    -float(rng.uniform(0.25, 0.55)), labeled=(i != 3)))
        # spot on top of graduation (normal structure overlap)
        defs.append(SynthDefect("black_spot", 0.02, float(height * 0.25 + 2 * max(8, height // 24)), 1.0, -0.4,
                                on_structure=True))
        defs.append(SynthDefect("blotch", float(rng.uniform(-math.pi, math.pi)), float(height * 0.55), 14.0, -0.05,
                                labeled=(b % 2 == 0)))
        defs.append(SynthDefect("bubble", float(rng.uniform(-math.pi, math.pi)), float(height * 0.4), 4.0, 0.25))
        defs.append(SynthDefect("black_spot", float(rng.uniform(-math.pi, math.pi)), float(height * 0.7), 1.0, -0.45,
                                vis_center=float(rng.uniform(0, 360)), vis_width=60.0))  # angle-dependent
        for k, d in enumerate(defs):
            d.uid = f"{s.code}-D{k + 1}"
        s.defects = defs
        specs.append(s)
    return specs


def generate_dataset(out_dir: str | Path, n_bottles: int = 4, n_angles: int = 36, width: int = 640, height: int = 512,
                     seed: int = 0, sessions: int = 1, label_every: int = 3) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    specs = default_bottles(n_bottles, width, height, seed)
    manifest = {"note": "SYNTHETIC data for software verification only; not representative of real optics.",
                "bottles": []}
    for spec in specs:
        tex, _ = _texture(spec, rng)
        for s in range(sessions):
            sdir = out / spec.code / f"S{s + 1:03d}"
            sdir.mkdir(parents=True, exist_ok=True)
            gt, labels = [], {"annotations": [], "normal_rois": [], "review_scopes": []}
            angles = [i * 360.0 / n_angles for i in range(n_angles)]
            with open(sdir / "angles.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["filename", "angle", "frame_index"])
                for i, a in enumerate(angles):
                    w.writerow([f"frame_{i:04d}.tif", f"{a:.3f}", i])
            for i, a in enumerate(angles):
                img = render(spec, a + (0.3 * s), rng, tex)
                tifffile.imwrite(sdir / f"frame_{i:04d}.tif", img)
                fr = {"frame": f"frame_{i:04d}.tif", "angle": a, "defects": []}
                for d in spec.defects:
                    p = project_defect(spec, d, a)
                    if p:
                        fr["defects"].append({"uid": d.uid, "kind": d.kind, "size": d.size, **p,
                                              "labeled": d.labeled})
                        if d.labeled and i % label_every == 0:
                            labels["annotations"].append({"frame": fr["frame"], "uid": d.uid, "kind": d.kind,
                                                          "x": p["x"], "y": p["y"], "size": d.size})
                gt.append(fr)
                if i % label_every == 0:
                    # a verified-normal ROI over graduation ticks (hard-negative style), where no defect is projected
                    xg = spec.width / 2.0 + spec.radius * math.sin(math.radians(a))
                    if math.cos(math.radians(a)) > 0.3:
                        roi = {"frame": fr["frame"], "class": "Normal Embossing",
                               "x0": xg - 30, "y0": height * 0.46, "x1": xg + 30, "y1": height * 0.6}
                        if not any(roi["x0"] - 4 <= dd["x"] <= roi["x1"] + 4 and roi["y0"] - 4 <= dd["y"] <= roi["y1"] + 4
                                   for dd in fr["defects"]):
                            labels["normal_rois"].append(roi)
            (sdir / "session_meta.json").write_text(json.dumps(
                {"camera_model": "SYNTH-CAM", "bit_depth": spec.bit_depth, "exposure_us": 1000.0, "gain_db": 0.0,
                 "lighting": "synthetic backlight", "lens": "synthetic", "magnification": 1.0, "mm_per_px": 0.02}),
                encoding="utf-8")
            (sdir / "ground_truth.json").write_text(json.dumps({"bottle": spec.code, "frames": gt,
                                                                "defects": [asdict(d) for d in spec.defects]}, indent=1),
                                                    encoding="utf-8")
            (sdir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
        manifest["bottles"].append({"code": spec.code, "product_type": spec.product_type, "bit_depth": spec.bit_depth})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


KIND_CLASS = {"black_spot": "Black Spot", "blotch": "Intensity / Transparency Defect", "bubble": "Bubble"}


def import_synthetic_labels(project, dataset_dir: str | Path) -> dict:
    """Load labels.json partial labels into the project (as if a human/customer had marked them)."""
    from .core.annotations import AnnotationStore
    store = AnnotationStore(project, author="synthetic-import")
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    ptype = {b["code"]: b["product_type"] for b in manifest["bottles"]}
    n = {"positives": 0, "normal_rois": 0}
    for b in project.bottles():
        if b["bottle_code"] in ptype:
            project.update_bottle(b["id"], product_type=ptype[b["bottle_code"]], verdict="ng")
        for s in project.sessions(b["id"]):
            lf = Path(s["source_dir"]) / "labels.json"
            if not lf.exists():
                continue
            lab = json.loads(lf.read_text(encoding="utf-8"))
            frames = {Path(f["path"]).name: f for f in project.frames(s["id"])}
            defect_ids = {}
            for a in lab["annotations"]:
                f = frames.get(a["frame"])
                if not f:
                    continue
                cls = KIND_CLASS[a["kind"]]
                if a["uid"] not in defect_ids:
                    defect_ids[a["uid"]] = store.create_defect(b["id"], cls, memo=a["uid"])
                if a["kind"] == "black_spot":
                    store.add(f["id"], cls, "positive", "point", {"x": a["x"], "y": a["y"], "r": max(1.0, a["size"])},
                              defect_id=defect_ids[a["uid"]], source="imported",
                              attributes={"size_px": math.pi * max(a["size"], 0.5) ** 2})
                else:
                    r = a["size"] * (1.5 if a["kind"] == "blotch" else 1.3)
                    store.add(f["id"], cls, "positive", "bbox",
                              {"x0": a["x"] - r, "y0": a["y"] - r, "x1": a["x"] + r, "y1": a["y"] + r},
                              defect_id=defect_ids[a["uid"]], source="imported")
                n["positives"] += 1
            for roi in lab["normal_rois"]:
                f = frames.get(roi["frame"])
                if f:
                    store.add(f["id"], roi["class"], "negative", "bbox",
                              {k: roi[k] for k in ("x0", "y0", "x1", "y1")}, source="imported")
                    n["normal_rois"] += 1
    return n
