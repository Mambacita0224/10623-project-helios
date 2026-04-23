# Helios-I2V Warm-Start Repair via PEFT

CMU 10-423/623/723 Generative AI, Spring 2026. Yuhang Zeng, Lucas Qin, Mark Pindur.

Private fork of [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios). We'll apply LoRA /
QLoRA to Helios-Distilled to fix the weak-motion warm-start failure mode in image-to-video
generation. Implementation lands in subsequent commits — this initial commit is the baseline.

## Layout

```
helios/, scripts/, eval/, tools/, example/    # upstream Helios (unchanged)
infer_helios.py, train_helios.py, install.sh  # upstream (unchanged)
UPSTREAM_README.md                            # original Helios README

notebooks/helios_trial.ipynb                  # Colab pilot baseline
modal/                                        # Modal cloud-GPU runner (baseline I2V/T2V)
```

## Baseline

- **Colab**: open `notebooks/helios_trial.ipynb` on an A100 40GB High-RAM runtime.
- **Modal** (preferred going forward): see [`modal/README.md`](./modal/README.md).

Modal workspace: `ac-Sb4ljxqqyGDfev8CNC7QVw`. AWS fallback: account `491325670598`, `us-east-1`.
