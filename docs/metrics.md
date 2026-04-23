# Evaluation Metrics

The metrics used to evaluate the warm-start fine-tune. Every metric is
**objective** (no human rating, no LLM-as-judge), **reproducible** (fixed
resolution, frame rate, and seed), and has an explicit numerical pass band.

## 1. What we measure

Hypothesis: Helios-Distilled suffers from a weak warm-start in image-to-video
(I2V) — the first ~1 second of generated frames contains substantially less
motion than (a) the same prompt under T2V or (b) the same clip's own later
frames. The metrics split into:

- **Primary** — directly target the warm-start failure mode.
- **Secondary** — sanity / side-effect checks (don't regress overall quality).

All thresholds are defined at the Helios inference default of **384 × 640**,
**16 fps**, 99-frame clips. Rescale thresholds linearly with `min(H, W)` when
evaluating at a different resolution (same convention as VBench).

## 2. Primary metrics

### 2.1 Early Motion Ratio (EMR)

Single scalar per clip operationalizing "how quickly does motion start".

```
EMR(V) = mean_{t in [1..K]}   flow_mag(V_{t-1}, V_t)
       / mean_{t in [K+1..N-1]} flow_mag(V_{t-1}, V_t)
```

- `flow_mag(a, b)` = mean magnitude of the Farneback dense optical-flow field
  between consecutive frames, same parameters as
  [`eval/1_get_motion_amplitude.py`](../eval/1_get_motion_amplitude.py)
  (`pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
  poly_sigma=1.2`).
- `N` = total frames in the clip (99 at 16 fps for the diagnostic).
- `K = 24` ≈ first 1.5 s ≈ one Helios chunk.

**Range.** `[0, +inf)`. Static first second → EMR → 0. Uniform motion across
the clip → EMR ≈ 1. Typical naturally-filmed web clips: 0.8 – 1.3.

**Pass threshold.** EMR ≥ 0.7.

### 2.2 Time-to-First-Motion (TTFM)

Index of the first frame whose flow magnitude exceeds the "clearly moving"
threshold.

```
TTFM(V) = min { t in [1..N-1] : per_frame_flow_mean(V_{t-1}, V_t) > tau }
          or N   (never reaches threshold)
tau     = 3.0 * min(H, W) / 256
```

Same Farneback as EMR, but mean-reduced across the full frame (no width-16
downscale). `tau` scaling matches VBench `StaticFilter`.

**Range.** `[0, N]`. **At 384 × 640: `tau ≈ 4.5` px/frame.**

**Pass threshold.** TTFM ≤ 6 (≈ 0.25 s at 16 fps).

## 3. Secondary metrics

### 3.1 Motion Amplitude

Script: [`eval/1_get_motion_amplitude.py`](../eval/1_get_motion_amplitude.py)
(unmodified). Mean Farneback flow magnitude (width-16 downscaled) across all
`N-1` frame pairs — the average motion magnitude of the whole clip.

**Range.** `[0, ~20+]`. **Pass threshold.** No hard cutoff; report the paired
delta vs baseline. Flag if the fine-tuned model increases EMR but drops
MotionAmp below baseline by more than 15 % — suggests a "moves early, freezes
late" failure.

### 3.2 Motion Smoothness

Script: [`eval/2_get_motion_smoothness.py`](../eval/2_get_motion_smoothness.py)
(unmodified). Drops every other frame, reinterpolates with AMT-S, and
reports `(255 - mean_abs_diff) / 255` between reinterpolated and original
middle frames.

**Range.** `[0, 1]`, higher = smoother. **Pass threshold.** ≥ 0.85.

Dependency: AMT-S checkpoint (~300 MB), downloaded via
[`eval/checkpoints/get_checkpoints.sh`](../eval/checkpoints/get_checkpoints.sh).

### 3.3 Drifting Motion Smoothness

Script:
[`eval/6_get_drifting_motion_smoothness.py`](../eval/6_get_drifting_motion_smoothness.py)
(unmodified). With `DRIFT_RATIO = 0.15`:

```
drift = | motion_smoothness(frames[:ceil(0.15*N)])
        - motion_smoothness(frames[floor(0.85*N):]) |
```

**Range.** `[0, 1]`, lower = stable quality across time. **Pass threshold.**
≤ 0.05.

## 4. Summary table

| Metric | Range | "Quick" | "Natural" | Script |
| --- | --- | --- | --- | --- |
| EMR | `[0, +inf)` | **≥ 0.7** | — | [`eval/compute_emr_ttfm.py`](../eval/compute_emr_ttfm.py) |
| TTFM | `[0, N]` frames | **≤ 6** | — | [`eval/compute_emr_ttfm.py`](../eval/compute_emr_ttfm.py) |
| Motion Amplitude | `[0, ~20+]` | — | Δ vs baseline | `eval/1_get_motion_amplitude.py` |
| Motion Smoothness | `[0, 1]` | — | **≥ 0.85** | `eval/2_get_motion_smoothness.py` |
| Drifting Smoothness | `[0, 1]` | — | **≤ 0.05** | `eval/6_get_drifting_motion_smoothness.py` |

A clip that passes both "Quick" thresholds *and* both "Natural" thresholds
is a success. Headline aggregates reported per method: paired mean Δ and
paired-t p-value against the zero-shot I2V baseline across the suite.

## 5. What we drop

- **Human ratings** — rating protocol not set up within our timeline.
- **GPT-based naturalness** (`eval/4_get_naturalness.py`) — orthogonal to the
  warm-start failure mode; adds a GPT dependency we don't want in the final
  report.
- **Aesthetic score** (`eval/0_get_aesthetic.py`) — orthogonal to motion
  quality.
- **VBench** — requires ~10-15 GB of extra checkpoints (RAFT + DINOv1 + ViCLIP
  + AMT) and a separate prompt-alignment step. Helios's in-repo
  motion/smoothness scripts give the same signal. Noted as a limitation.

## 6. Reproducibility

- Inference resolution: **384 × 640** (Helios default).
- Frame rate: **16 fps**, clip length: **99 frames** (≈ 6.2 s).
- Seed: **0** for the diagnostic; three fixed seeds `{0, 1, 2}` for the
  fine-tune-vs-baseline comparison.
- For each metric we report per-clip scores (JSON), mean ± std over the
  25-prompt suite, and the paired-t statistic + p-value for every
  method-vs-baseline comparison.
- `K = 24` is fixed. If `num_latent_frames_per_chunk` changes, update K in
  the same commit.

## 7. Observed baseline numbers (n = 25, 99-frame clips at 384 × 640)

Raw data: [`outputs/diagnostic/emr_ttfm_results.json`](../outputs/diagnostic/emr_ttfm_results.json).
Figure: [`outputs/diagnostic/early_motion_figure.png`](../outputs/diagnostic/early_motion_figure.png).

### 7.1 Aggregate per condition

| Condition | EMR (↑) | TTFM frames (↓) | Motion Amplitude |
| --- | --- | --- | --- |
| **T2V** (zero-shot, no image history) | **0.865 ± 0.393** | 82.6 ± 29.6 | 1.28 |
| **I2V** (zero-shot) | 0.509 ± 0.354 | 94.8 ± 6.2 | 0.60 |
| **I2V + amp** (`--is_amplify_first_chunk`) | 0.591 ± 0.372 | 94.8 ± 5.8 | 0.61 |

- EMR ≥ 0.7: **T2V passes** (0.865); **I2V fails** (0.509); **I2V+amp fails** (0.591).
- TTFM ≤ 6: all three conditions fail — Helios produces a gentle motion ramp
  rather than an instant on-switch. EMR carries the headline claim (see §7.3).

### 7.2 Paired comparisons (prompt-matched)

| Comparison | ΔEMR | t | p (two-sided) | Cohen's *d* |
| --- | --- | --- | --- | --- |
| I2V − T2V | **−0.355** | −3.94 | **0.0001** | ≈ 0.96 |
| (I2V + amp) − T2V | −0.274 | −2.67 | 0.0077 | ≈ 0.73 |

Adding an image history to a text prompt reduces first-1.5 s motion magnitude
by ~41 % (p = 0.0001, n = 25), confirming the warm-start pathology with a
large effect size. Helios's built-in `--is_amplify_first_chunk` recovers only
≈ +0.08 EMR (0.509 → 0.591), still failing the "Quick" threshold and still
significantly below T2V (p = 0.0077). That residual gap motivates the PEFT
approach.

### 7.3 Interpretation notes

- **TTFM ceiling effect.** At τ = 4.5 px/frame, almost every I2V clip fails
  to cross the threshold before the clip ends, so TTFM saturates near 95. EMR
  is the primary metric going forward; TTFM is reported as a secondary data
  point.
- **Motion Amplitude halves** (T2V 1.28 → I2V 0.60, −53 %). The bug isn't
  just a delayed start; the whole clip moves less. The fine-tune needs to
  improve both EMR (when motion starts) and MotionAmp (how much sustains).
- **Absolute amplitudes sit below the ≥ 3 band in §3.1.** Helios-Distilled
  at 384 × 640 is intrinsically smoother than raw web video; interpret
  numbers as relative deltas vs T2V rather than against the web-video
  reference.
- **Figure.** T2V is roughly flat at ~1.2 px/frame from frame 1 onward; I2V
  and I2V+amp start near 0 and take ~3 s to ramp toward T2V's level. The
  K = 24 cutoff sits squarely in the flat region of the I2V curve.
