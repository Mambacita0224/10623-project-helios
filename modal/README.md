# Running the Helios baseline on Modal

Modal workspace: `ac-Sb4ljxqqyGDfev8CNC7QVw`.

## One-time setup (per developer machine)

```bash
pip install modal
modal setup                                       # browser auth; picks the right workspace
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx
modal volume create helios-models                 # caches the ~20 GB Helios-Distilled weights
```

## Run the baseline

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

First run builds the container (~5–10 min) and downloads weights (~20 GB, one-time). Subsequent runs
cold-start in ~30 s and only pay for GPU seconds.

## GPU selection

Edit the `gpu=` kwarg on `_run_infer` in `app.py`:

- `"A100-80GB"` — default, matches Colab pilot.
- `"H100"` — ~2x faster, best for sweeps.
- `"L40S"` (48 GB) — cheaper, needs `--enable_low_vram_mode --group_offloading_type leaf_level` in
  the Helios CLI args.
