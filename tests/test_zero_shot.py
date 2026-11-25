"""Zero-shot models: real WinCLIP (Anomalib) and OpenCLIP inference. Skipped when weights are not cached locally
(run setup.ps1 -DownloadModels first). Marked slow."""
import os

import numpy as np
import pytest

from gistudio.algorithms.zero_shot.common import MODEL_DIR, configure_model_cache, resolution_warnings

configure_model_cache()
HAS_WEIGHTS = (MODEL_DIR / "clip_cache" / "vit_b_16_plus_240-laion400m_e31-8fb26589.pt").exists()


def test_resolution_warnings_report_stride():
    w, info = resolution_warnings(480, 240, 16)
    assert info["resize_factor"] == 0.5 and info["effective_stride_original_px"] == 32
    assert any("DOWNSCALED" in x for x in w) and any("NOT defect probabilities" in x for x in w)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_WEIGHTS, reason="WinCLIP backbone weights not downloaded")
def test_winclip_tiled_inference_runs():
    os.environ["HF_HUB_OFFLINE"] = "1"   # must run from local cache: no network, no image upload
    from gistudio.algorithms.base import create
    img = np.full((300, 420), 2500, np.uint16)
    img[150:153, 200:203] = 300
    res = create("zeroshot.winclip", tile=240, overlap=40, batch=4).run(img, {"bit_depth": 12})
    m = res.score_maps["winclip_score"]
    assert m.shape == img.shape and np.isfinite(m).all()
    assert res.model_info["k_shot"] == 0 and res.model_info["effective_stride_original_px"] == 16


@pytest.mark.slow
@pytest.mark.skipif(not HAS_WEIGHTS, reason="CLIP weights not downloaded")
def test_clip_patch_prompts_run():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from gistudio.algorithms.base import create
    img = np.full((256, 256), 2500, np.uint16)
    res = create("zeroshot.clip_patch", tile=128, overlap=0).run(img, {"bit_depth": 12})
    assert res.score_maps["clip_defect_prob"].shape == img.shape
    probs = res.model_info["mean_prompt_prob"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3
