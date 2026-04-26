# Fine-Tune Methodology for Helios I2V Warm-Start Repair

This document describes the full methodology used in this repository to fine-tune Helios-Distilled for image-to-video (I2V) warm-start behavior, based on the actual implementation and configs in this project.

It is intended as a teammate-facing, end-to-end reference from data pipeline to checkpoint selection.

## 1. Problem definition and target behavior

### 1.1 Observed failure mode

In zero-shot I2V, Helios-Distilled often starts with weak motion in early frames, then ramps later. This causes:

- weak first-second dynamics
- delayed motion onset
- unstable quality when trying aggressive decode settings

### 1.2 Practical objective

Improve temporal response in early frames while preserving late-frame motion and overall clip quality.

Concretely, we optimize for:

- stronger early motion (warm-start)
- no late-frame collapse
- no severe motion-amplitude regression

## 2. System architecture used in this repo

### 2.1 Compute split

- Local machine:
  - controls runs
  - stores docs/results
  - runs CPU evaluation scripts
- Modal cloud:
  - training and inference on GPU
  - persistent volumes for models and curated dataset

### 2.2 Main code paths

- Data curation: `tools/prepare_mixkit.py`
- Latent export: `tools/mixkit_export_latents.py`
- Training launcher: `modal/app.py::train_mixkit_lora`
- Training core: `train_helios.py`
- Diagnostic generation: `modal/diagnostic.py`
- Metric/evaluation:
  - `eval/compute_emr_ttfm.py`
  - `eval/smoke_gate.py`

### 2.3 Persistent storage

- Modal volume `helios-models`: model checkpoints cache
- Modal volume `helios-mixkit`: curated clips, manifests, latents, train outputs

## 3. Data pipeline (implemented workflow)

## 3.1 Source dataset and filtering

We use Mixkit-Src and curate motion-oriented clips via `tools/prepare_mixkit.py`.

Implemented curation defaults:

- target fps: 24
- clip length: 99 frames
- target resolution: 384x640
- flow method: Farneback
- early window: K=24
- EMR keep-band: [0.7, 1.3]
- early flow minimum: 2.5

The script writes:

- curated H.264 clips
- first-frame PNG
- JSONL manifest (`manifest.jsonl` on volume, pulled locally as `data/train/mixkit_curated.jsonl`)

Typical execution:

```bash
modal run tools/prepare_mixkit.py::download_source
modal run tools/prepare_mixkit.py::build_manifest
modal run tools/prepare_mixkit.py::pull_manifest
```

## 3.2 Latent export for stage-1 training

Training consumes precomputed `.pt` latent payloads (not raw MP4 decoding in train loop).

`tools/mixkit_export_latents.py` encodes each clip into:

- `vae_latent`
- `prompt_embed`
- `first_frames_image`
- `prompt_raw`

Critical implementation detail for 99-frame training:

- use `--pixel_frames_per_section 33`
- this splits 99 frames into 3 sections
- saved latent shape becomes `[3, C, 9, H, W]`
- aligns with stage-1 latent window/chunk assumptions

Execution through Modal wrapper:

```bash
modal run modal/app.py::export_mixkit_latents \
  --jsonl /vol/mixkit_curated/manifest.jsonl \
  --video-root /vol/mixkit_curated \
  --out-dir /vol/mixkit_curated/latents_pt_99 \
  --max-frames 99
```

Optional integrity audit:

```bash
modal run modal/app.py::audit_mixkit_latents \
  --jsonl /vol/mixkit_curated/manifest.jsonl \
  --out-dir /vol/mixkit_curated/latents_pt_99
```

## 4. Training strategy

## 4.1 Stage and objective

We train stage-1 style LoRA adapters for warm-start repair.

Key settings used in configs:

- `is_enable_stage1: true`
- `latent_window_size: [9]`
- `min_num_frame: 99`
- i2v-focused random drop curriculum

## 4.2 LoRA scope and capacity

Implemented scope is selective (early blocks, self-attn only):

- controlled by `lora_early_attn1_blocks`
- targets `blocks[0..B-1].attn1` (`to_q`, `to_k`, `to_v`, `to_out.0`)

This is narrower and safer than full-model-wide LoRA for preserving semantics.

## 4.3 Validation-mode improvements added in code

To keep validation aligned with stable inference behavior, this repo adds:

- `validation_force_stage2` in `helios/utils/train_config.py`
- logic in `train_helios.py` to enable stage-2 during validation even if training is stage-1
- I2V validation support via:
  - `validation_sample_type`
  - `validation_image_path`
  - `validation_image_noise_sigma_min/max`

This allows stable validation decode without changing the training objective.

## 4.4 Configs and their role

- Smoke run config:
  - `scripts/training/configs/mixkit_lora_smoke_modal.yaml`
  - short run for quick model-quality signal
- Full run config:
  - `scripts/training/configs/mixkit_lora_modal.yaml`
  - longer run after smoke confirms direction
- Light continuation from best smoke checkpoint:
  - `scripts/training/configs/mixkit_lora_light_from50_modal.yaml`
  - starts from `/vol/mixkit_curated/train_lora_smoke/checkpoint-50`

## 4.5 Launch commands

Smoke:

```bash
modal run modal/app.py::train_mixkit_lora \
  --config scripts/training/configs/mixkit_lora_smoke_modal.yaml
```

Full:

```bash
modal run modal/app.py::train_mixkit_lora \
  --config scripts/training/configs/mixkit_lora_modal.yaml
```

Light continuation:

```bash
modal run modal/app.py::train_mixkit_lora \
  --config scripts/training/configs/mixkit_lora_light_from50_modal.yaml
```

## 4.6 Practical training caveat

`train_helios.py` enforces config consistency by checking `output_dir/config.json`.
If YAML changes but old config exists, training aborts with mismatch.

Fix command:

```bash
modal run modal/app.py::clear_mixkit_train_stale_config \
  --path /vol/mixkit_curated/train_lora/config.json
```

For smoke/light runs, use the corresponding output path.

## 5. Inference and diagnostic evaluation protocol

## 5.1 Diagnostic generation

`modal/diagnostic.py` runs the 25-prompt suite and writes videos under `outputs/diagnostic`.

Two inference profiles are implemented:

- `emr`: historical baseline profile for warm-start analysis
- `mixkit`: profile aligned with training-time validation behavior

For this project, `mixkit` profile is used for checkpoint comparison to keep train/eval consistency.

## 5.2 Stable I2V decode settings used

In project practice, stable comparison runs use:

- 99 frames
- latent chunk size 9
- stage2 enabled with `[2, 2, 2]`
- guidance scale 1.0
- image noise sigma range 0.111 to 0.135
- fps 24

Example command pattern:

```bash
modal run modal/diagnostic.py::run_all \
  --conditions i2v_lora \
  --inference-profile mixkit \
  --lora-path <checkpoint>/pytorch_lora_weights.safetensors \
  --overwrite
```

## 5.3 Metrics used

Primary and gate metrics are computed through:

- `eval/compute_emr_ttfm.py`
- `eval/smoke_gate.py`

Smoke gate command pattern:

```bash
conda run -n helios-dev python eval/smoke_gate.py \
  --baseline-dir outputs/diagnostic/mixkit/i2v \
  --candidate-dir outputs/diagnostic/mixkit/<candidate_dir> \
  --fps 24
```

Current gate defaults in implementation (`eval/smoke_gate.py`):

- min early absolute: 0.18
- min late absolute: 0.45
- min TTFM: 90.0
- min early ratio vs baseline: 0.70
- min late ratio vs baseline: 0.70
- min motion amplitude ratio vs baseline: 0.70

## 6. Checkpoint selection methodology used here

## 6.1 Why we compare multiple checkpoints

Warm-start repair can improve early metrics at one step and regress quality later. Therefore, we do checkpoint-wise diagnostics, not just final-step evaluation.

## 6.2 Observed checkpoint outcomes (from50 continuation)

Compared candidates:

- `i2v_lora_ckpt50` (PASS)
- `i2v_lora_from50_ckpt100` (PASS, strongest overall)
- `i2v_lora_from50_ckpt200` (FAIL)

Key observation:

- ckpt200 shows high EMR but late-motion and amplitude collapse
- smoke gate catches this as FAIL
- ckpt100 gives strongest balanced gains and is preferred

## 6.3 Decision rule in practice

We prioritize checkpoints that satisfy all of:

- smoke gate PASS
- improved early and late motion ratios vs baseline
- improved motion amplitude ratio vs baseline
- no visible semantic/motion collapse in diagnostic videos

## 7. Recommended reproducible runbook

1. Prepare Mixkit curated manifest and clips.
2. Export latents to `/vol/mixkit_curated/latents_pt_99`.
3. Run smoke training config.
4. Generate candidate diagnostics with `inference-profile mixkit`.
5. Run smoke gate against fixed baseline `outputs/diagnostic/mixkit/i2v`.
6. Compare multiple checkpoints and choose best balanced one.
7. Continue with light/full training from that checkpoint.
8. Re-evaluate and update selection.

## 8. Typical failure modes and fixes

- Missing checkpoint path:
  - ensure `load_model_path` points to existing folder, e.g. `checkpoint-50` not `checkpoint-50-final`.
- Config mismatch error before training starts:
  - remove stale output `config.json` via `clear_mixkit_train_stale_config`.
- EMA-related environment issue:
  - if deepspeed is unavailable in Modal image, keep `use_ema: false`.
- Inconsistent train/eval behavior:
  - keep 99/9 alignment and use validation/inference settings that match the stable mixkit recipe.

## 9. Deliverables generated by this methodology

- Curated training manifest: `data/train/mixkit_curated.jsonl`
- Latent dataset on volume: `/vol/mixkit_curated/latents_pt_99`
- Checkpoints on volume under train output directories
- Diagnostic videos under `outputs/diagnostic/...`
- Smoke gate JSON/figures under `outputs/diagnostic/...`
- Team docs:
  - `docs/metrics.md`
  - `docs/methodology.md`

## 10. Scope and limitations

- This workflow is tuned for Helios-Distilled at 384x640, 24 fps, 99 frames.
- Results are highly configuration-sensitive; preserving the exact decode profile matters.
- Current evaluation emphasizes objective motion metrics; human preference studies are out of scope for this course timeline.
