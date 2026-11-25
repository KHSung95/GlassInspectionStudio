"""Download pretrained weights from their OFFICIAL distribution and record source / version / license / hash.

    python tools/download_models.py

open_clip (pinned to the range declared by Anomalib: >=2.23.0,<2.26.1) downloads the WinCLIP backbone
ViT-B-16-plus-240 / laion400m_e31 from the official open_clip GitHub release. Weights are stored in
<repo>/models/clip_cache (see gistudio.algorithms.zero_shot.common.configure_model_cache) and the registry is
written to models/model_registry.json. Only model files are downloaded; no image or project data is uploaded.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = [
    {"id": "winclip-backbone", "used_by": ["zeroshot.winclip", "zeroshot.clip_patch"],
     "open_clip_model": "ViT-B-16-plus-240", "pretrained": "laion400m_e31"},
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from gistudio.algorithms.zero_shot.common import MODEL_DIR, configure_model_cache
    configure_model_cache()
    import open_clip
    from importlib.metadata import version
    from open_clip import pretrained as opt
    reg = {"generated_at": datetime.now().isoformat(timespec="seconds"), "models": []}
    for m in MODELS:
        cfg = opt.get_pretrained_cfg(m["open_clip_model"], m["pretrained"])
        open_clip.create_model_and_transforms(m["open_clip_model"], pretrained=m["pretrained"])
        url = cfg.get("url", "")
        f = MODEL_DIR / "clip_cache" / Path(url).name
        digest = sha256(f) if f.exists() else None
        expected = Path(url).stem.split("-")[-1]  # open_clip embeds a sha256 prefix in the filename
        entry = {**m, "source": url, "source_kind": "official open_clip GitHub release",
                 "local_file": str(f.relative_to(REPO)) if f.exists() else None, "size": f.stat().st_size if f.exists() else None,
                 "sha256": digest, "sha256_prefix_expected": expected,
                 "sha256_verified": bool(digest and digest.startswith(expected)),
                 "open_clip_version": version("open_clip_torch"), "anomalib_version": version("anomalib"),
                 "license": "open_clip code: MIT (https://github.com/mlfoundations/open_clip/blob/main/LICENSE). "
                            "Weights: released by the open_clip/LAION authors, trained on LAION-400M — review the "
                            "LAION dataset terms before commercial deployment.",
                 "anomalib_license": "Apache-2.0"}
        reg["models"].append(entry)
        print(json.dumps(entry, indent=1))
    out = MODEL_DIR / "model_registry.json"
    out.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"registry -> {out}")


if __name__ == "__main__":
    main()
