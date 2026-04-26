"""Encode Mixkit `manifest.jsonl` / `mixkit_curated.jsonl` clips to stage-1 `.pt` latents.

Matches the on-disk format produced by `tools/offload_data/get_long-latents.py`:
  {uttid}_{T}_{H}_{W}.pt
with `vae_latent`, `prompt_embed`, `first_frames_image` (PIL), `prompt_raw`.

After export, set `data_config.min_num_frame` to the same T (e.g. 99) and point
`instance_data_root` at the output folder. Delete `dataset_cache.pkl` in that
folder or use `force_rebuild: true` if you re-export with different T.

Example (local, clips under video_root):
  python tools/mixkit_export_latents.py \\
    --jsonl data/train/mixkit_curated.jsonl \\
    --video_root /path/to/mixkit_curated \\
    --out_dir outputs/mixkit_latents_pt \\
    --pretrained_model_name_or_path /path/to/Helios-Distilled

On Modal (volume `helios-mixkit` mounted at `/vol`):
  --jsonl /vol/mixkit_curated/manifest.jsonl \\
  --video_root /vol/mixkit_curated \\
  --out_dir /vol/mixkit_curated/latents_pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel
from video_reader import PyVideoReader

# Run as `python tools/mixkit_export_latents.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helios.utils.utils_base import encode_prompt  # noqa: E402


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_video_tensor(
    path: str,
    target_h: int,
    target_w: int,
    max_frames: int,
) -> torch.Tensor:
    """Return (T, C, H, W) float in [-1, 1], resized to (target_h, target_w)."""
    vr = PyVideoReader(path, threads=0)
    vlen, _, _ = vr.get_shape()
    n = min(int(vlen), int(max_frames))
    if n < 1:
        raise ValueError(f"No frames: {path}")
    indices = list(range(n))
    frames = torch.from_numpy(vr.get_batch(indices)).float()
    frames = (frames / 127.5) - 1.0
    video = frames.permute(0, 3, 1, 2)  # T, C, H, W
    t, c, h, w = video.shape
    if (h, w) != (target_h, target_w):
        out = []
        for i in range(t):
            out.append(TF.resize(video[i], (target_h, target_w), antialias=True))
        video = torch.stack(out, dim=0)
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="mixkit_curated.jsonl or Modal manifest.jsonl",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Root that clip_path in each row is relative to (e.g. mixkit_curated/)",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="BestWishYsh/Helios-Distilled",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=99,
        help="Max frames to read per clip (Mixkit prep defaults to 99).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip if target .pt already exists (same as get_long-latents skip).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (same as upstream offload scripts).")
    device = torch.device("cuda:0")
    weight_dtype = torch.bfloat16

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, weight_dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, weight_dtype)
    )

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    rows = _load_jsonl(args.jsonl)
    video_root = os.path.abspath(args.video_root)
    n_ok = 0
    n_skip = 0

    for rec in tqdm(rows, desc="mixkit export"):
        uttid = str(rec["id"])
        clip_rel = rec["clip_path"]
        video_path = os.path.join(video_root, clip_rel)
        prompt = rec["prompt"]

        if not os.path.isfile(video_path):
            print(f"[skip] missing video: {video_path}", file=sys.stderr)
            continue

        try:
            video = _load_video_tensor(
                video_path, args.height, args.width, max_frames=args.max_frames
            )
        except Exception as e:
            print(f"[skip] read failed {video_path}: {e}", file=sys.stderr)
            continue

        t = video.shape[0]
        if t < 1:
            continue

        # Filename must list *pixel* T for bucketing (dataloader history latents).
        out_name = f"{uttid}_{t}_{args.height}_{args.width}.pt"
        out_path = os.path.join(args.out_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        with torch.no_grad():
            pixel_values = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(
                device=device, dtype=vae.dtype
            )
            vae_latents = vae.encode(pixel_values).latent_dist.sample()
            vae_latents = (vae_latents - latents_mean) * latents_std
            if vae_latents.dim() == 4:
                vae_latents = vae_latents.unsqueeze(0)
            if vae_latents.dim() != 5:
                raise RuntimeError(
                    f"Expected 5D vae latents (b,c,t,h,w), got shape {tuple(vae_latents.shape)}"
                )

            pe, _pam = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt=[prompt],
                device=device,
            )

        image_tensor = (video[0:1] + 1) / 2 * 255
        image_tensor = image_tensor.to(torch.uint8)
        first_pil = transforms.ToPILImage()(image_tensor[0])

        payload = {
            "vae_latent": vae_latents.cpu().detach(),
            "prompt_embed": pe[0].cpu().detach(),
            "first_frames_image": first_pil,
            "prompt_raw": prompt,
        }
        try:
            torch.save(payload, out_path)
        except Exception as e:
            print(f"[skip] save failed {out_path}: {e}", file=sys.stderr)
            continue

        n_ok += 1
        free_memory()

    print(f"Done. wrote {n_ok} files to {args.out_dir} (skipped existing: {n_skip})")


if __name__ == "__main__":
    main()
