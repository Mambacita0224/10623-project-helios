"""Modal entrypoint for running Helios-Distilled inference.

Run locally with, e.g.:

    modal run modal/app.py::t2v_sanity_check
    modal run modal/app.py::i2v --image-path example/wave.jpg --prompt "..."

The first run will build the image and download the Helios-Distilled weights into
the `helios-models` Modal Volume (created once with `modal volume create helios-models`).
Subsequent runs reuse the cached image + weights.

See ``modal/README.md`` for setup.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "helios-i2v-peft"
REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root on the local machine
MODEL_DIR = "/root/models"  # where we mount the persistent volume inside the container
MODEL_NAME = "BestWishYsh/Helios-Distilled"


# ---------------------------------------------------------------------------
# Image: Helios deps on top of a CUDA 12.6 + PyTorch 2.10 base
# ---------------------------------------------------------------------------
# The upstream install.sh pins torch==2.10.0 / cu126. We do the same here and
# layer the remaining pip deps on top. We also vendor the repo into /root/helios.

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
        "opencv-python",
        "moviepy",
        "imageio-ffmpeg",
        "ftfy",
        "Jinja2",
        "einops",
        "omegaconf",
        "loguru",
        "packaging",
        "ninja",
    )
    # Mount the repo so infer_helios.py / helios/ / scripts/ are importable at /root/helios
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

# Persistent volume that caches Helios checkpoints across runs.
models_volume = modal.Volume.from_name("helios-models", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# One-time download helper (run on first use of any generation function)
# ---------------------------------------------------------------------------
def _ensure_model_downloaded(repo_id: str = MODEL_NAME) -> str:
    """Download `repo_id` into the persistent volume if it isn't there yet."""
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
    Path(marker).touch()
    # Persist volume changes
    models_volume.commit()
    return local_dir


# ---------------------------------------------------------------------------
# Helios inference function (single GPU)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",  # change to "H100" for ~2x speed or "L40S" (48GB) w/ group_offloading
    timeout=60 * 60,  # 1 hour cap per call
    volumes={MODEL_DIR: models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def _run_infer(cli_args: list[str], model_repo: str = MODEL_NAME) -> bytes:
    """Run infer_helios.py with the given CLI args, return the generated MP4 as bytes."""
    import subprocess
    import sys

    local_model_dir = _ensure_model_downloaded(model_repo)

    # Helios expects to be run from its own root so relative paths (example/wave.jpg) resolve.
    os.chdir("/root/helios")

    # Prepend the model-path args if the caller didn't supply them.
    has_base = any(a.startswith("--base_model_path") for a in cli_args)
    has_tx = any(a.startswith("--transformer_path") for a in cli_args)
    if not has_base:
        cli_args = ["--base_model_path", local_model_dir] + cli_args
    if not has_tx:
        cli_args = ["--transformer_path", local_model_dir] + cli_args

    output_folder = "/tmp/helios_out"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    has_out = any(a.startswith("--output_folder") for a in cli_args)
    if not has_out:
        cli_args = cli_args + ["--output_folder", output_folder]

    cmd = [sys.executable, "infer_helios.py", *cli_args]
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Pick up the most-recently modified .mp4 produced by Helios.
    candidates = sorted(Path(output_folder).rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("No .mp4 produced; check logs above.")
    mp4_path = candidates[-1]
    print(f"Returning {mp4_path} ({mp4_path.stat().st_size / 1e6:.1f} MB)")
    return mp4_path.read_bytes()


# ---------------------------------------------------------------------------
# Local entrypoints — these are what `modal run` invokes.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def t2v_sanity_check(
    num_frames: int = 99,
    output_path: str = "outputs/t2v_sanity.mp4",
):
    """Generate a short T2V clip to verify the environment works end to end."""
    prompt = (
        "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, "
        "turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive "
        "orange spot on its side, its fins moving fluidly. Close-up shot with dynamic movement."
    )
    args = [
        "--sample_type", "t2v",
        "--prompt", prompt,
        "--num_frames", str(num_frames),
        "--guidance_scale", "1.0",
        "--is_enable_stage2",
        "--pyramid_num_inference_steps_list", "2", "2", "2",
        "--is_amplify_first_chunk",
    ]
    mp4_bytes = _run_infer.remote(args)
    _save_bytes(mp4_bytes, output_path)


@app.local_entrypoint()
def i2v(
    image_path: str,
    prompt: str,
    num_frames: int = 99,
    guidance_scale: float = 1.0,
    image_noise_sigma_min: float = 0.111,
    image_noise_sigma_max: float = 0.135,
    is_skip_first_chunk: bool = False,
    output_path: str = "outputs/i2v.mp4",
):
    """Run Helios-Distilled I2V on a single image+prompt pair.

    ``image_path`` is resolved relative to the repo root inside the container (so
    ``example/wave.jpg`` works out of the box). If you want to pass a local image
    that isn't in the repo, drop it into ``example/`` first.
    """
    args = [
        "--sample_type", "i2v",
        "--image_path", image_path,
        "--prompt", prompt,
        "--num_frames", str(num_frames),
        "--fps", "24",
        "--guidance_scale", str(guidance_scale),
        "--image_noise_sigma_min", str(image_noise_sigma_min),
        "--image_noise_sigma_max", str(image_noise_sigma_max),
        "--is_enable_stage2",
        "--pyramid_num_inference_steps_list", "2", "2", "2",
        "--is_amplify_first_chunk",
    ]
    if is_skip_first_chunk:
        args.append("--is_skip_first_chunk")
    mp4_bytes = _run_infer.remote(args)
    _save_bytes(mp4_bytes, output_path)


# ---------------------------------------------------------------------------
# Helper: write remote bytes to the local filesystem.
# ---------------------------------------------------------------------------
def _save_bytes(data: bytes, local_path: str) -> None:
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    print(f"Saved {local} ({local.stat().st_size / 1e6:.1f} MB)")
