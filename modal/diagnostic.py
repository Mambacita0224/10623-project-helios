"""Modal entrypoint: run the 25-prompt diagnostic on Helios-Distilled.

Three conditions per prompt:

    1. ``t2v``      — text-to-video (no image history).
    2. ``i2v``      — image-to-video, zero-shot.
    3. ``i2v_amp``  — I2V with ``--is_amplify_first_chunk``.

(2) and (3) feed the SDXL-Turbo-generated image at
``data/diagnostic/images/{id}.png`` (see ``modal/prepare_images.py``).
Outputs land under ``outputs/diagnostic/{condition}/{id}.mp4``.

Usage:

    modal run modal/prepare_images.py::generate_all      # first-frame images
    modal run modal/diagnostic.py::run_all               # full 75-run suite
    modal run modal/diagnostic.py::run_all --limit 1     # smoke test
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
from typing import List, Optional

import modal

APP_NAME = "helios-diagnostic"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = "/root/models"
MODEL_NAME = "BestWishYsh/Helios-Distilled"

# Image recipe matches modal/app.py (inlined since `modal/` shadows the SDK
# package name, making a cross-file import awkward).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "triton==3.6.0",
        "kernels==0.13.0",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers==5.3.0",
        "sentence-transformers==5.2.3",
        "accelerate==1.12.0",
        "peft==0.18.1",
        "huggingface-hub[cli]==1.4.1",
        "zstandard==0.25.0",
        "video-reader-rs==0.4.1",
        "numpy<2.0.0",
        "pandas",
        "opencv-python",
        "moviepy",
        "imageio-ffmpeg",
        "ftfy",
        "regex",
        "Jinja2",
        "einops",
        "omegaconf",
        "loguru",
        "packaging",
        "ninja",
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/root/helios", ignore=[
        "**/.git/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/outputs/**",
        "**/output_helios/**",
        "**/BestWishYsh/**",
        "**/*.mp4",
        "**/*.safetensors",
        "**/*.ckpt",
        "**/*.pdf",
    ])
)

models_volume = modal.Volume.from_name("helios-models", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


# ---- Remote helpers ---------------------------------------------------------

def _ensure_model_downloaded(repo_id: str = MODEL_NAME) -> str:
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(MODEL_DIR, repo_id)
    marker = os.path.join(local_dir, ".download_complete")
    if os.path.exists(marker):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        max_workers=8,
    )
    pathlib.Path(marker).touch()
    models_volume.commit()
    return local_dir


@app.function(
    gpu="A100-80GB",
    timeout=60 * 30,
    volumes={MODEL_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=8,  # fan-out cap for .starmap (workspace A100 quota = 10)
)
def run_one(
    condition: str,
    clip_id: str,
    prompt: str,
    image_b64: Optional[str],
    num_frames: int,
    guidance_scale: float,
) -> dict:
    """Run Helios inference under one condition.

    Returns {"condition", "clip_id", "mp4_bytes"} so the caller can route the
    result to the correct output path regardless of completion order.
    """
    import subprocess
    import sys

    local_model_dir = _ensure_model_downloaded()
    os.chdir("/root/helios")

    out_dir = "/tmp/helios_out"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    args = [
        "--base_model_path", local_model_dir,
        "--transformer_path", local_model_dir,
        "--prompt", prompt,
        "--num_frames", str(num_frames),
        "--guidance_scale", str(guidance_scale),
        "--is_enable_stage2",
        "--pyramid_num_inference_steps_list", "2", "2", "2",
        "--output_folder", out_dir,
    ]
    if condition == "t2v":
        args = ["--sample_type", "t2v", *args]
    elif condition in ("i2v", "i2v_amp"):
        if not image_b64:
            raise RuntimeError(f"condition {condition} requires an input image")
        image_path = "/tmp/helios_input.png"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        args = [
            "--sample_type", "i2v",
            "--image_path", image_path,
            "--fps", "24",
            "--image_noise_sigma_min", "0.111",
            "--image_noise_sigma_max", "0.135",
            *args,
        ]
        if condition == "i2v_amp":
            args.append("--is_amplify_first_chunk")
    else:
        raise ValueError(f"unknown condition: {condition}")

    cmd = [sys.executable, "infer_helios.py", *args]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    candidates = sorted(
        pathlib.Path(out_dir).rglob("*.mp4"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise RuntimeError("no .mp4 produced; check logs above")
    mp4_bytes = candidates[-1].read_bytes()
    print(f"returning {candidates[-1].name} ({len(mp4_bytes) / 1e6:.1f} MB)")
    return {"condition": condition, "clip_id": clip_id, "mp4_bytes": mp4_bytes}


# ---- Local driver -----------------------------------------------------------

def _load_prompts(path: pathlib.Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@app.local_entrypoint()
def run_all(
    conditions: str = "t2v,i2v,i2v_amp",
    num_frames: int = 99,
    guidance_scale: float = 1.0,
    overwrite: bool = False,
    limit: int = 0,
):
    """Run the diagnostic on every prompt x every requested condition.

    ``num_frames=99`` matches Helios's example config and keeps the EMR
    "late" window (frames 25..98) large enough to be a stable reference.
    Use ``--limit N`` to smoke-test before committing to the full suite.
    """
    prompts_path = REPO_ROOT / "data" / "diagnostic" / "prompts.jsonl"
    images_dir = REPO_ROOT / "data" / "diagnostic" / "images"
    out_root = REPO_ROOT / "outputs" / "diagnostic"

    entries = _load_prompts(prompts_path)
    if limit > 0:
        entries = entries[:limit]

    cond_list = [c.strip() for c in conditions.split(",") if c.strip()]
    for c in cond_list:
        if c not in {"t2v", "i2v", "i2v_amp"}:
            raise SystemExit(f"unknown condition: {c!r}")
        (out_root / c).mkdir(parents=True, exist_ok=True)

    inputs = []
    for e in entries:
        for c in cond_list:
            out = out_root / c / f"{e['id']}.mp4"
            if out.exists() and not overwrite:
                print(f"[skip] {c}/{e['id']}.mp4 already exists")
                continue
            image_b64 = None
            if c in ("i2v", "i2v_amp"):
                ip = images_dir / f"{e['id']}.png"
                if not ip.exists():
                    raise SystemExit(
                        f"missing image for {e['id']}; run modal/prepare_images.py first"
                    )
                image_b64 = base64.b64encode(ip.read_bytes()).decode("ascii")
            inputs.append((c, e["id"], e["prompt"], image_b64, num_frames, guidance_scale))

    if not inputs:
        print("nothing to do; pass --overwrite to regenerate")
        return

    import time as _time
    print(
        f"\nfanning out {len(inputs)} run(s) across up to 8 parallel "
        f"A100-80GB containers ...\n"
    )
    start = _time.time()
    completed = 0
    for result in run_one.starmap(inputs, order_outputs=False):
        cond = result["condition"]
        clip_id = result["clip_id"]
        mp4_bytes = result["mp4_bytes"]
        out = out_root / cond / f"{clip_id}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(mp4_bytes)
        completed += 1
        elapsed = _time.time() - start
        print(
            f"[{completed}/{len(inputs)}] {cond}/{clip_id}.mp4  "
            f"({len(mp4_bytes) / 1e6:.1f} MB)  elapsed {elapsed:.0f}s"
        )

    total = _time.time() - start
    print(f"\ndone; clips saved under {out_root} in {total:.0f}s "
          f"({total / 60:.1f} min)")
    print(
        "\nnext: compute metrics:\n"
        "    python eval/compute_emr_ttfm.py \\\n"
        "        --conditions t2v=outputs/diagnostic/t2v "
        "i2v=outputs/diagnostic/i2v i2v_amp=outputs/diagnostic/i2v_amp \\\n"
        "        --output outputs/diagnostic/emr_ttfm_results.json \\\n"
        "        --figure outputs/diagnostic/early_motion_figure.png"
    )
