"""Generate first-frame images for the diagnostic prompt suite.

Reads `data/diagnostic/prompts.jsonl`, generates one 768x432 image per entry
with SDXL-Turbo (4 steps, CFG=0), and writes PNGs to
`data/diagnostic/images/{id}.png`. Idempotent; pass `--overwrite` to force
regeneration. All images run in a single container (one cold start).

    modal run modal/prepare_images.py::generate_all
"""

from __future__ import annotations

import io
import json
import pathlib
from typing import List, Tuple

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_NAME = "helios-diagnostic-images"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "diffusers==0.32.2",
        "transformers==4.48.0",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "Pillow",
        "huggingface-hub==0.28.1",
    )
)

app = modal.App(APP_NAME, image=image)


@app.cls(gpu="A10G", timeout=60 * 30)
class SdxlTurbo:
    @modal.enter()
    def _load(self):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.torch = torch

    @modal.method()
    def generate_batch(self, specs: List[Tuple[str, str, int]], width: int,
                       height: int) -> List[Tuple[str, bytes]]:
        """specs = [(id, prompt, seed), ...].  Returns [(id, png_bytes), ...]."""
        out = []
        for pid, prompt, seed in specs:
            generator = self.torch.Generator(device="cuda").manual_seed(int(seed))
            img = self.pipe(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=width,
                height=height,
                generator=generator,
            ).images[0]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append((pid, buf.getvalue()))
            print(f"[gen] {pid}  ({len(buf.getvalue()) // 1024} KB)")
        return out


@app.local_entrypoint()
def generate_all(overwrite: bool = False, width: int = 768, height: int = 432):
    prompts_path = REPO_ROOT / "data" / "diagnostic" / "prompts.jsonl"
    images_dir = REPO_ROOT / "data" / "diagnostic" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with prompts_path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    specs: List[Tuple[str, str, int]] = []
    for e in entries:
        out = images_dir / f"{e['id']}.png"
        if out.exists() and not overwrite:
            print(f"[skip] {out.name} already exists")
            continue
        specs.append((e["id"], e["image_prompt"], int(e["id"])))

    if not specs:
        print("all images already generated; nothing to do")
        return

    print(f"generating {len(specs)} image(s) in one container...")
    results = SdxlTurbo().generate_batch.remote(specs, width, height)
    for pid, png_bytes in results:
        (images_dir / f"{pid}.png").write_bytes(png_bytes)
    print(f"\ndone; images at {images_dir}")
