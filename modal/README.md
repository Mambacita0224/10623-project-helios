# Helios on Modal: environment setup and baseline

## 1. Local developer environment

Use a dedicated venv or conda env so the `modal` CLI and local tooling (eval scripts, Hugging
Face CLI, notebook, small OpenCV jobs) are isolated. Heavy GPU work runs **in Modal
containers**; their dependencies are **not** in `requirements-dev.txt`—they are declared inline
in `modal/app.py` and `tools/prepare_mixkit.py`.

From the [requirements-dev.txt](../requirements-dev.txt) header:

```bash
# from repo root 10623-project-helios
conda create -n helios-dev python=3.11 -y
conda activate helios-dev
pip install --upgrade pip
pip install -r requirements-dev.txt
```

That file pins **modal** (job dispatch), **opencv** / **numpy** / **matplotlib** / **tqdm** (local
eval, plots), **huggingface_hub[cli]** (optional local HF downloads), and **jupyter** /
**ipykernel** (e.g. `notebooks/helios_trial.ipynb`).

```bash
modal profile current   # should show a workspace; if empty, run: modal token new
# or:  modal setup
```

## 2. One-time Modal setup (per developer machine)

```bash
pip install modal   # already satisfied if you used requirements-dev.txt
modal setup                                      # browser auth; picks the right workspace
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx   # use --force to overwrite
modal volume create helios-models                # caches the ~20 GB Helios-Distilled weights
```

### Weights & Biases (Mixkit LoRA training on Modal)

`train_mixkit_lora` uses `scripts/training/configs/mixkit_lora_modal.yaml` with `report_to: wandb`. Create a **Modal** secret (name must match `modal/app.py`: `wandb`) so the container receives `WANDB_API_KEY`. Do **not** commit API keys; paste the value only in the shell command.

```bash
# One-time. Optional: add WANDB_ENTITY=your-team WANDB_PROJECT=helios-mixkit
modal secret create wandb WANDB_API_KEY=xxxxxxxxxxxxxxxx

# If the secret name already exists and you need to replace it:
modal secret create wandb WANDB_API_KEY=xxxxxxxxxxxxxxxx --force
```

If a key is ever exposed (chat, screen share), revoke it in the [W&B user settings](https://wandb.ai/settings) and create a new one, then re-run `modal secret create` with `--force`.

**Mixkit LoRA (poster) scope** — `scripts/training/configs/mixkit_lora_modal.yaml` uses `lora_early_attn1_blocks` (attn1-only, early blocks), `use_early_latent_time_loss_weight`, and rank 16. After you change PEFT/loss fields, clear stale `output_dir/config.json` on the volume: `modal run modal/app.py::clear_mixkit_train_stale_config` (or use a separate `output_dir` like the smoke file below).

**Quick smoke (100 steps, plan-aligned, separate W&B + checkpoint dir):**

`modal run modal/app.py::train_mixkit_lora --config scripts/training/configs/mixkit_lora_smoke_modal.yaml`

Validation during training is **I2V** by default in `mixkit_lora*_modal.yaml` (`validation_sample_type: i2v`, `example/wave.jpg`). Put other stills under a path in the repo and point `validation_image_path` at them, or set `validation_sample_type: t2v` for text-only. `train_helios` now builds the validation `HeliosPipeline` with the checkpoint’s **`subfolder=scheduler`** (HeliosScheduler), same as `infer_helios`—**not** the UniPC scheduler used for flow training—so I2V validation should match a normal local inference run.

`download_source` in `tools/prepare_mixkit.py` also mounts this secret so large HuggingFace
dataset pulls are less likely to hit **HTTP 429** (see main README / Mixkit section below).

## 3. Run the Helios baseline (inference on Modal)

```bash
cd 10623-project-helios

# T2V smoke test (99 frames, ~2–3 min on A100-80GB)
modal run modal/app.py::t2v_sanity_check

# I2V on the upstream wave example — matches our Colab pilot
modal run modal/app.py::i2v \
    --image-path example/wave.jpg \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave's body." \
    --num-frames 99 \
    --output-path outputs/wave_baseline.mp4
```

The first run builds the container (~5–10 min) and downloads weights (~20 GB, one-time on the
`helios-models` volume). Later runs cold-start in ~30 s; you only pay for GPU time.

**GPU selection** — edit the `gpu=` kwarg on `_run_infer` in `app.py`:

- `"A100-80GB"` — default, matches Colab pilot
- `"H100"` — faster sweeps
- `"L40S"` (48 GB) — cheaper; pass `--enable_low_vram_mode --group_offloading_type leaf_level` in the Helios CLI args

## 4. Mixkit training data (PEFT / curated clips)

`huggingface-secret` is **required** for `download_source`. Without `HF_TOKEN`, full dataset
pulls can return **HTTP 429**.

```bash
cd 10623-project-helios
modal run tools/prepare_mixkit.py::download_source
# optional: verify ProcessPool + OpenCV (~1 min)
modal run tools/prepare_mixkit.py::smoke_process_pool
# full manifest (parallel workers; override with e.g. --num-workers 12)
modal run tools/prepare_mixkit.py::build_manifest
modal run tools/prepare_mixkit.py::pull_manifest
```

Volume: `helios-mixkit`. The manifest is written locally to `data/train/mixkit_curated.jsonl`.

Curated clips are encoded as **H.264** (yuv420p) via `ffmpeg` so they open in macOS Preview.
If you still have older **`mp4v`** files from an earlier run, re-encode locally:
`bash tools/reencode_mp4_h264_local.sh outputs/mixkit_qc30` (requires `brew install ffmpeg`).

