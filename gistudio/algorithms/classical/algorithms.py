"""Classical vision baselines. Outputs are CANDIDATES, never ground truth.

Spot detectors express their score as an SNR: response / robust noise sigma (unit-free, comparable across
bit depths). Area detectors express deviation in units of robust noise sigma as well.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..base import Candidate, InspectionAlgorithm, InspectionResult, ParamSpec, register
from . import ops

SNR_SEMANTICS = "SNR: detector response divided by robust image noise sigma (not a probability)"


def _spot_candidates(alg: InspectionAlgorithm, img: np.ndarray, score: np.ndarray, thr: float,
                     class_name: str, min_area: int, max_area: int, max_count: int) -> list[Candidate]:
    out = []
    for c in ops.components(score, thr, min_area, max_area, max_count):
        lc = ops.local_contrast_at(img, c["peak_x"], c["peak_y"])
        out.append(Candidate(
            frame_id=None, model_id=f"{alg.algorithm_id}@{alg.version}", source_algorithm=alg.algorithm_id,
            class_name=class_name, score=c["score"], geometry_type="point",
            geometry={"x": c["peak_x"], "y": c["peak_y"], "r": max(1.5, float(np.sqrt(c["area"] / np.pi)))},
            area_px=c["area"], features={"bbox": c["bbox"], "local_contrast": lc["contrast"],
                                         "outer_mean": lc["outer"]}))
    return out


def _area_candidates(alg: InspectionAlgorithm, score: np.ndarray, thr: float, class_name: str,
                     min_area: int, max_count: int) -> list[Candidate]:
    out = []
    for c in ops.components(score, thr, min_area, 10**9, max_count):
        out.append(Candidate(
            frame_id=None, model_id=f"{alg.algorithm_id}@{alg.version}", source_algorithm=alg.algorithm_id,
            class_name=class_name, score=c["score"], geometry_type="bbox",
            geometry={"x0": c["bbox"][0], "y0": c["bbox"][1], "x1": c["bbox"][2], "y1": c["bbox"][3]},
            area_px=c["area"], features={"peak": (c["peak_x"], c["peak_y"])}))
    return out


COMMON_SPOT = [
    ParamSpec("threshold", "float", 6.0, 0.5, 100.0, 0.5, help="SNR threshold"),
    ParamSpec("min_area", "int", 1, 1, 10000),
    ParamSpec("max_area", "int", 400, 1, 100000),
    ParamSpec("max_candidates", "int", 500, 1, 100000),
    ParamSpec("pre_sigma", "float", 0.0, 0.0, 5.0, 0.1, help="Optional Gaussian pre-smoothing (px)"),
]


# =========================================================================== black spot detectors
@register
class BlackHatDetector(InspectionAlgorithm):
    algorithm_id = "classical.blackhat"
    display_name = "Black-hat morphology"
    score_semantics = SNR_SEMANTICS
    params_spec = [ParamSpec("kernel", "int", 9, 3, 101, 2, help="Structuring element diameter (px) > spot size")] \
        + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        s = ops.blackhat(x, p["kernel"]) / sig
        return InspectionResult(_spot_candidates(self, image, s, p["threshold"], "Black Spot", p["min_area"],
                                                 p["max_area"], p["max_candidates"]),
                                {"blackhat_snr": s}, preprocess={"pre_sigma": p["pre_sigma"], "noise_sigma": sig})


@register
class DoGDetector(InspectionAlgorithm):
    algorithm_id = "classical.dog"
    display_name = "Difference of Gaussian"
    score_semantics = SNR_SEMANTICS
    params_spec = [ParamSpec("sigma_small", "float", 1.0, 0.3, 10, 0.1),
                   ParamSpec("sigma_large", "float", 4.0, 0.5, 50, 0.5)] + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        s = ops.dog_dark(x, p["sigma_small"], p["sigma_large"]) / sig
        return InspectionResult(_spot_candidates(self, image, s, p["threshold"], "Black Spot", p["min_area"],
                                                 p["max_area"], p["max_candidates"]),
                                {"dog_snr": s}, preprocess={"noise_sigma": sig})


@register
class LoGBlobDetector(InspectionAlgorithm):
    algorithm_id = "classical.log_blob"
    display_name = "Multi-scale LoG blob"
    score_semantics = SNR_SEMANTICS + "; scale-normalised LoG"
    params_spec = [ParamSpec("sigma_min", "float", 0.8, 0.3, 20, 0.1),
                   ParamSpec("sigma_max", "float", 3.0, 0.3, 50, 0.1),
                   ParamSpec("n_scales", "int", 4, 1, 12)] + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        sigmas = list(np.geomspace(p["sigma_min"], max(p["sigma_max"], p["sigma_min"]), int(p["n_scales"])))
        r, smap = ops.log_dark_multiscale(x, sigmas)
        s = r / sig
        cands = _spot_candidates(self, image, s, p["threshold"], "Black Spot", p["min_area"], p["max_area"],
                                 p["max_candidates"])
        for c in cands:
            ix, iy = int(c.geometry["x"]), int(c.geometry["y"])
            c.features["blob_sigma"] = float(smap[iy, ix])
            c.geometry["r"] = float(smap[iy, ix] * np.sqrt(2))
        return InspectionResult(cands, {"log_snr": s}, preprocess={"noise_sigma": sig, "sigmas": sigmas})


@register
class LocalBackgroundDetector(InspectionAlgorithm):
    algorithm_id = "classical.local_background"
    display_name = "Local background subtraction"
    score_semantics = SNR_SEMANTICS + "; (median background - image)"
    params_spec = [ParamSpec("bg_kernel", "int", 31, 5, 501, 2),
                   ParamSpec("downsample", "int", 2, 1, 16)] + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        bg = ops.background_median(x, p["bg_kernel"], p["downsample"])
        s = (bg - x) / sig
        return InspectionResult(_spot_candidates(self, image, s, p["threshold"], "Black Spot", p["min_area"],
                                                 p["max_area"], p["max_candidates"]),
                                {"bg_sub_snr": s, "background": bg}, preprocess={"noise_sigma": sig})


@register
class LocalContrastDetector(InspectionAlgorithm):
    algorithm_id = "classical.local_contrast"
    display_name = "Local contrast (z-score)"
    score_semantics = "Local z-score: (local mean - pixel) / (local std + noise sigma)"
    params_spec = [ParamSpec("window", "int", 15, 3, 301, 2)] + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        m, sd = ops.local_mean_std(x, p["window"])
        s = (m - x) / (sd + sig)
        return InspectionResult(_spot_candidates(self, image, s, p["threshold"] / 2.0, "Black Spot", p["min_area"],
                                                 p["max_area"], p["max_candidates"]),
                                {"local_z": s}, preprocess={"noise_sigma": sig},
                                warnings=["threshold is halved for z-score scale"])


@register
class AdaptiveThresholdDetector(InspectionAlgorithm):
    algorithm_id = "classical.adaptive_threshold"
    display_name = "Adaptive threshold + CCA"
    score_semantics = "(local mean - pixel) / noise sigma, masked by adaptive threshold"
    params_spec = [ParamSpec("window", "int", 25, 3, 301, 2),
                   ParamSpec("offset_sigma", "float", 5.0, 0.5, 100, 0.5, help="pixel < mean - offset*sigma")] \
        + COMMON_SPOT

    def _run(self, image, meta):
        p = self.params
        x = ops.gauss(ops.f32(image), p["pre_sigma"])
        sig = ops.noise_sigma_meta(image, meta)
        m, _ = ops.local_mean_std(x, p["window"])
        s = (m - x) / sig
        s = np.where(s >= p["offset_sigma"], s, 0).astype(np.float32)
        return InspectionResult(_spot_candidates(self, image, s, max(p["threshold"], p["offset_sigma"]), "Black Spot",
                                                 p["min_area"], p["max_area"], p["max_candidates"]),
                                {"adaptive_snr": s}, preprocess={"noise_sigma": sig})


# =========================================================================== intensity / transparency
@register
class LowFrequencyDeviation(InspectionAlgorithm):
    algorithm_id = "classical.lowfreq_deviation"
    display_name = "Multi-scale low-frequency deviation"
    default_class = "Intensity / Transparency Defect"
    score_semantics = "|band-pass deviation| / noise sigma, max over scales (both polarities)"
    params_spec = [ParamSpec("scales", "str", "4,8,16", help="comma separated sigma list (px)"),
                   ParamSpec("context_factor", "float", 4.0, 1.5, 20, 0.5),
                   ParamSpec("threshold", "float", 3.0, 0.2, 100, 0.1),
                   ParamSpec("min_area", "int", 50, 1, 10**7),
                   ParamSpec("max_candidates", "int", 200, 1, 100000)]

    def _run(self, image, meta):
        p = self.params
        x = ops.f32(image)
        sig = ops.noise_sigma_meta(image, meta)
        scales = [float(s) for s in str(p["scales"]).split(",") if s.strip()]
        best = np.zeros_like(x)
        for s in scales:
            d = np.abs(ops.gauss(x, s) - ops.gauss(x, s * p["context_factor"])) / (sig / max(1.0, s))
            # noise of a Gaussian-averaged signal shrinks ~ 1/s, hence the normalisation above
            best = np.maximum(best, d)
        return InspectionResult(_area_candidates(self, best, p["threshold"], self.default_class, p["min_area"],
                                                 p["max_candidates"]), {"lowfreq_dev": best},
                                preprocess={"noise_sigma": sig, "scales": scales})


@register
class LocalIntensityDeviation(InspectionAlgorithm):
    algorithm_id = "classical.local_intensity_deviation"
    display_name = "Local intensity deviation"
    default_class = "Intensity / Transparency Defect"
    score_semantics = "|smoothed - background| / smoothed-noise sigma"
    params_spec = [ParamSpec("smooth_sigma", "float", 3.0, 0.5, 30, 0.5),
                   ParamSpec("bg_kernel", "int", 101, 9, 1001, 2),
                   ParamSpec("threshold", "float", 4.0, 0.2, 100, 0.1),
                   ParamSpec("min_area", "int", 50, 1, 10**7),
                   ParamSpec("max_candidates", "int", 200, 1, 100000)]

    def _run(self, image, meta):
        p = self.params
        x = ops.f32(image)
        sig = ops.noise_sigma_meta(image, meta) / max(1.0, p["smooth_sigma"])
        sm = ops.gauss(x, p["smooth_sigma"])
        bg = ops.background_median(sm, p["bg_kernel"], 4)
        s = np.abs(sm - bg) / sig
        return InspectionResult(_area_candidates(self, s, p["threshold"], self.default_class, p["min_area"],
                                                 p["max_candidates"]), {"intensity_dev": s, "background": bg},
                                preprocess={"noise_sigma_smoothed": sig})


@register
class GradientTexture(InspectionAlgorithm):
    algorithm_id = "classical.gradient_texture"
    display_name = "Gradient / texture anomaly"
    default_class = "Intensity / Transparency Defect"
    score_semantics = "local std ratio vs. wide-context median (dimensionless)"
    params_spec = [ParamSpec("window", "int", 15, 3, 201, 2),
                   ParamSpec("context", "int", 151, 15, 1001, 2),
                   ParamSpec("threshold", "float", 3.0, 1.0, 50, 0.1),
                   ParamSpec("min_area", "int", 30, 1, 10**7),
                   ParamSpec("max_candidates", "int", 200, 1, 100000)]

    def _run(self, image, meta):
        p = self.params
        x = ops.f32(image)
        sig = ops.noise_sigma_meta(image, meta)
        _, sd = ops.local_mean_std(x, p["window"])
        ref = ops.background_median(sd, p["context"], 4)
        s = (sd + sig) / (ref + sig)
        return InspectionResult(_area_candidates(self, s, p["threshold"], self.default_class, p["min_area"],
                                                 p["max_candidates"]), {"texture_ratio": s,
                                                                         "gradient": ops.gradient_mag(x)})


# =========================================================================== linear / broken voids
@register
class RidgeVoidDetector(InspectionAlgorithm):
    """Thin elongated structures (linear or boomerang-like broken internal voids, chains of tiny voids).

    Multi-scale Hessian ridge strength (largest |eigenvalue| across scales, scale-normalised) for dark and/or
    bright lines, then connected components filtered by elongation. Score = peak ridge strength / noise sigma.
    """
    algorithm_id = "classical.ridge_void"
    display_name = "Linear / broken void (Hessian ridge)"
    default_class = "Linear / Broken Void"
    score_semantics = "scale-normalised Hessian ridge strength / noise sigma (not a probability)"
    params_spec = [ParamSpec("sigma_min", "float", 1.0, 0.5, 20, 0.5),
                   ParamSpec("sigma_max", "float", 3.0, 0.5, 40, 0.5),
                   ParamSpec("n_scales", "int", 3, 1, 8),
                   ParamSpec("polarity", "choice", "both", choices=["dark", "bright", "both"]),
                   ParamSpec("threshold", "float", 8.0, 0.5, 200, 0.5),
                   ParamSpec("min_length", "float", 15.0, 1, 5000, 1, help="component major axis (px)"),
                   ParamSpec("min_elongation", "float", 3.0, 1.0, 50, 0.5, help="major / minor axis"),
                   ParamSpec("max_candidates", "int", 300, 1, 100000)]

    def _run(self, image, meta):
        p = self.params
        x = ops.f32(image)
        sig = ops.noise_sigma_meta(image, meta)
        sigmas = list(np.geomspace(p["sigma_min"], max(p["sigma_min"], p["sigma_max"]), int(p["n_scales"])))
        best = np.zeros_like(x)
        for s in sigmas:
            g = ops.gauss(x, s)
            dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
            tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
            l1 = 0.5 * (dxx + dyy + tmp)      # largest eigenvalue
            l2 = 0.5 * (dxx + dyy - tmp)      # smallest eigenvalue
            # dark line on bright background -> large positive l1 across the line; bright line -> negative l2
            r = np.zeros_like(x)
            if p["polarity"] in ("dark", "both"):
                r = np.maximum(r, np.where(np.abs(l1) > 2 * np.abs(l2), l1, 0))
            if p["polarity"] in ("bright", "both"):
                r = np.maximum(r, np.where(np.abs(l2) > 2 * np.abs(l1), -l2, 0))
            best = np.maximum(best, (s * s) * r / 4.0)
        score = best / sig
        mask = (score >= p["threshold"]).astype(np.uint8)
        out = []
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 5:
                continue
            ys, xs = np.nonzero(lab[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
                                    stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]] == i)
            cov = np.cov(np.stack([xs, ys]).astype(np.float64)) if len(xs) > 2 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            major, minor = 4 * np.sqrt(max(ev[0], 1e-6)), 4 * np.sqrt(max(ev[1], 1e-6))
            elong = major / max(minor, 1.0)
            if major < p["min_length"] or elong < p["min_elongation"]:
                continue
            x0, y0 = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
            w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            sc = float(score[y0:y0 + h, x0:x0 + w][lab[y0:y0 + h, x0:x0 + w] == i].max())
            out.append(Candidate(None, f"{self.algorithm_id}@{self.version}", self.algorithm_id, self.default_class, sc,
                                 "bbox", {"x0": float(x0), "y0": float(y0), "x1": float(x0 + w), "y1": float(y0 + h)},
                                 area_px=area, features={"length": float(major), "elongation": float(elong)}))
        out.sort(key=lambda c: -c.score)
        return InspectionResult(out[: p["max_candidates"]], {"ridge_snr": score}, preprocess={"noise_sigma": sig,
                                                                                               "sigmas": sigmas})


# =========================================================================== bubbles / shape
@register
class BubbleContourDetector(InspectionAlgorithm):
    algorithm_id = "classical.bubble_contour"
    display_name = "Bubble: blob + contour + ring analysis"
    default_class = "Bubble"
    score_semantics = "circularity x |ring contrast| / noise sigma (heuristic)"
    params_spec = [ParamSpec("sigma_min", "float", 2.0, 0.5, 30, 0.5),
                   ParamSpec("sigma_max", "float", 8.0, 1.0, 80, 0.5),
                   ParamSpec("blob_threshold", "float", 4.0, 0.5, 100, 0.5),
                   ParamSpec("min_circularity", "float", 0.5, 0.0, 1.0, 0.05),
                   ParamSpec("canny_low", "float", 2.0, 0.5, 50, 0.5, help="in noise-sigma units"),
                   ParamSpec("canny_high", "float", 5.0, 1.0, 100, 0.5),
                   ParamSpec("threshold", "float", 3.0, 0.0, 100, 0.5),
                   ParamSpec("max_candidates", "int", 200, 1, 100000)]

    def _run(self, image, meta):
        p = self.params
        x = ops.f32(image)
        sig = ops.noise_sigma_meta(image, meta)
        sigmas = list(np.geomspace(p["sigma_min"], p["sigma_max"], 4))
        # both polarities: bubbles show bright and dark rims
        r_dark, s_dark = ops.log_dark_multiscale(x, sigmas)
        r_bright, s_bright = ops.log_dark_multiscale(-x, sigmas)
        resp = np.maximum(r_dark, r_bright) / sig
        smap = np.where(r_dark >= r_bright, s_dark, s_bright)
        # edges for contour analysis (scaled to 8-bit relative to noise)
        g = ops.gauss(x, 1.0)
        lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
        g8 = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        k = 255.0 / max(hi - lo, 1e-6) * sig
        edges = cv2.Canny(g8, p["canny_low"] * k, p["canny_high"] * k)
        out = []
        for c in ops.components(resp, p["blob_threshold"], 1, 10**9, 5 * p["max_candidates"]):
            cx, cy = c["peak_x"], c["peak_y"]
            rad = float(smap[int(cy), int(cx)] * np.sqrt(2)) + 1
            R = int(rad * 2) + 3
            x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
            x1, y1 = min(x.shape[1], int(cx) + R + 1), min(x.shape[0], int(cy) + R + 1)
            e = edges[y0:y1, x0:x1].copy()
            cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            circ, ell = 0.0, None
            for cnt in cnts:
                a, per = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
                if per > 0 and len(cnt) >= 5:
                    ci = 4 * np.pi * a / (per * per)
                    if ci > circ:
                        circ = ci
                        (ex, ey), (ma, mi), ang = cv2.fitEllipse(cnt)
                        ell = (ex + x0, ey + y0, ma, mi, ang)
            inner = ops.local_contrast_at(x, cx, cy, r_in=max(1.0, rad * 0.5), r_out=rad * 1.8)
            ring = abs(inner["contrast"]) / sig
            # blob without closed contour still gets a reduced circularity prior
            circ_eff = circ if circ > 0 else 0.3
            if circ > 0 and circ < p["min_circularity"]:
                continue
            score = circ_eff * ring
            if score < p["threshold"]:
                continue
            out.append(Candidate(None, f"{self.algorithm_id}@{self.version}", self.algorithm_id, "Bubble", score,
                                 "bbox", {"x0": cx - rad, "y0": cy - rad, "x1": cx + rad, "y1": cy + rad},
                                 area_px=float(np.pi * rad * rad),
                                 features={"circularity": circ, "ring_snr": ring, "ellipse": ell,
                                           "polarity": "dark" if inner["contrast"] > 0 else "bright"}))
        out.sort(key=lambda c: -c.score)
        return InspectionResult(out[: p["max_candidates"]], {"bubble_log_snr": resp,
                                                             "edges": edges.astype(np.float32)},
                                preprocess={"noise_sigma": sig})
