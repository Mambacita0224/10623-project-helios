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
        # train_helios.py uses StatefulDataLoader (torchdata); not part of the base torch wheel.
        "torchdata",
        "tensorboard",  # optional; some configs use report_to: tensorboard
        "wandb==0.23.0",  # mixkit_lora_modal.yaml uses report_to: wandb + Modal secret `wandb`
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
# Mixkit curated manifest + clips (from tools/prepare_mixkit.py on app `helios-mixkit-prep`).
MIXKIT_MOUNT = "/vol"
mixkit_volume = modal.Volume.from_name("helios-mixkit", create_if_missing=True)

app = modal.App(APP_NAME, image=image)

# Training: keep in sync — request N GPUs here and pass the same N to accelerate.
# Examples: "A100-80GB" (1 GPU), "A100-80GB:2" (2× DDP), "H100:2" (faster, if available).
_TRAIN_GPU = "A100-80GB:4"
_TRAIN_NUM_PROCESSES = 4
_TRAIN_CONFIG = "scripts/training/configs/mixkit_lora_modal.yaml"


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
# Mixkit → stage-1 latent export (VAE + text encoder, writes .pt on volume)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={MODEL_DIR: models_volume, MIXKIT_MOUNT: mixkit_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def _export_mixkit_latents_remote(
    jsonl: str = "/vol/mixkit_curated/manifest.jsonl",
    video_root: str = "/vol/mixkit_curated",
    out_dir: str = "/vol/mixkit_curated/latents_pt",
    model_repo: str = MODEL_NAME,
    max_frames: int = 49,
    skip_existing: bool = True,
) -> str:
    """VAE+UMT5 encode Mixkit clips from jsonl; write .pt for stage-1 / train_helios.

    Set ``data_config.min_num_frame`` to the same as ``max_frames`` (e.g. 49) and
    ``instance_data_root: [\"<out_dir>\"]`` with ``use_stage1_dataset: true``.
    """
    import subprocess
    import sys

    local_model_dir = _ensure_model_downloaded(model_repo)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir("/root/helios")
    cmd = [
        sys.executable,
        "tools/mixkit_export_latents.py",
        "--jsonl",
        jsonl,
        "--video_root",
        video_root,
        "--out_dir",
        out_dir,
        "--pretrained_model_name_or_path",
        local_model_dir,
        "--max_frames",
        str(max_frames),
    ]
    if skip_existing:
        cmd.append("--skip_existing")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    mixkit_volume.commit()
    return f"ok: {out_dir}"


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
# Mixkit LoRA training (Accelerate DDP, checkpoints on `helios-mixkit` volume)
# ---------------------------------------------------------------------------
@app.function(
    gpu=_TRAIN_GPU,
    timeout=60 * 60 * 12,
    cpu=8,
    volumes={MODEL_DIR: models_volume, MIXKIT_MOUNT: mixkit_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        # Create once: `modal secret create wandb WANDB_API_KEY=...` (see modal/README.md)
        modal.Secret.from_name("wandb"),
    ],
)
def _train_mixkit_lora_remote(
    num_processes: int = _TRAIN_NUM_PROCESSES,
    config_rel: str = _TRAIN_CONFIG,
) -> str:
    """Run ``train_helios.py`` with ``mixkit_lora_modal.yaml``; logs and checkpoints under ``/vol/mixkit_curated/train_lora``."""
    import subprocess
    import sys

    _ensure_model_downloaded(MODEL_NAME)
    os.makedirs("/vol/mixkit_curated", exist_ok=True)
    os.chdir("/root/helios")
    if not os.path.isabs(config_rel):
        cfg = os.path.join("/root/helios", config_rel)
    else:
        cfg = config_rel
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"config not found: {cfg}")
    if num_processes < 1:
        raise ValueError("num_processes must be >= 1")
    print(
        f">>> train_helios num_processes={num_processes} config={cfg}",
        flush=True,
    )
    cmd = [
        "accelerate",
        "launch",
        "--num_machines",
        "1",
        "--machine_rank",
        "0",
        "--num_processes",
        str(num_processes),
        "--mixed_precision",
        "bf16",
    ]
    if num_processes > 1:
        cmd.append("--multi_gpu")
    cmd.extend(["train_helios.py", "--config", cfg])
    subprocess.run(cmd, check=True, env={**os.environ, "ACCELERATE_LOG_LEVEL": "INFO"})
    mixkit_volume.commit()
    return f"ok: see output_dir in {config_rel} (e.g. /vol/mixkit_curated/train_lora)"


@app.function(
    timeout=120,
    volumes={MIXKIT_MOUNT: mixkit_volume},
)
def _remove_mixkit_train_config_json(
    path: str = "/vol/mixkit_curated/train_lora/config.json",
) -> str:
    """Delete stale ``config.json`` so a new YAML (e.g. after latent_window_size change) is accepted.

    ``train_helios`` refuses to start if ``output_dir/config.json`` from an older run disagrees
    with the current ``--config`` (see ``Configuration mismatch``). Run this once, then
    ``train_mixkit_lora`` again. Does not delete checkpoints; only remove whole ``train_lora/``
    if you need a full wipe.
    """
    if os.path.isfile(path):
        os.remove(path)
        mixkit_volume.commit()
        return f"removed {path}"
    return f"skip: {path} not found"


# ---------------------------------------------------------------------------
# Local entrypoints — these are what `modal run` invokes.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def clear_mixkit_train_stale_config(
    path: str = "/vol/mixkit_curated/train_lora/config.json",
):
    """One-shot: remove old ``config.json`` on the mixkit volume after you change the training YAML.

    Examples:

    - Full run (default): ``modal run modal/app.py::clear_mixkit_train_stale_config``
    - Smoke (``mixkit_lora_smoke_modal.yaml``): same with
      ``--path /vol/mixkit_curated/train_lora_smoke/config.json``
    """
    print(_remove_mixkit_train_config_json.remote(path=path))


@app.local_entrypoint()
def train_mixkit_lora(
    num_processes: int = _TRAIN_NUM_PROCESSES,
    config: str = _TRAIN_CONFIG,
):
    """Fine-tune LoRA on precomputed Mixkit latents (DDP on multiple GPUs if ``_TRAIN_GPU`` requests them).

    Requires: ``export_mixkit_latents`` has populated ``/vol/mixkit_curated/latents_pt`` on
    volume ``helios-mixkit``, and this app mounts the same volume at ``/vol``.

    W&B: create ``modal secret create wandb WANDB_API_KEY=...`` (see ``modal/README.md``).

    **Cost / speed:** edit ``_TRAIN_GPU`` and ``_TRAIN_NUM_PROCESSES`` at the top of ``modal/app.py``
    so the GPU count matches ``num_processes`` (e.g. ``A100-80GB`` + ``num_processes=1`` for cheaper runs).
    """
    msg = _train_mixkit_lora_remote.remote(
        num_processes=num_processes,
        config_rel=config,
    )
    print(msg)


@app.local_entrypoint()
def export_mixkit_latents(
    jsonl: str = "/vol/mixkit_curated/manifest.jsonl",
    video_root: str = "/vol/mixkit_curated",
    out_dir: str = "/vol/mixkit_curated/latents_pt",
    model_repo: str = MODEL_NAME,
    max_frames: int = 49,
    skip_existing: bool = True,
):
    """Offline encode Mixkit clips on GPU; writes Helios stage-1 `.pt` files to the volume.

    Requires prior ``modal run tools/prepare_mixkit.py::build_manifest`` so the
    manifest and `clips/*.mp4` exist under ``/vol/mixkit_curated`` on ``helios-mixkit``.
    """
    msg = _export_mixkit_latents_remote.remote(
        jsonl=jsonl,
        video_root=video_root,
        out_dir=out_dir,
        model_repo=model_repo,
        max_frames=max_frames,
        skip_existing=skip_existing,
    )
    print(msg)


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
