"""Compute EMR and TTFM from diagnostic-run MP4s and plot the time series.

Input: one or more `name=dir` pairs, each dir containing one MP4 per prompt
(e.g. `001.mp4`, `002.mp4`, ...). Output: a JSON with per-clip scores,
per-condition aggregates, and paired-t results against ``t2v``; plus a PNG
of mean per-frame flow vs time (± 1 std) with the K=24 cutoff line.

Farneback parameters and formulas match docs/metrics.md. CPU only.

Usage:

    python eval/compute_emr_ttfm.py \\
        --conditions t2v=outputs/diagnostic/t2v \\
                     i2v=outputs/diagnostic/i2v \\
                     i2v_amp=outputs/diagnostic/i2v_amp \\
        --output outputs/diagnostic/emr_ttfm_results.json \\
        --figure outputs/diagnostic/early_motion_figure.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np


# Must match docs/metrics.md / eval/1_get_motion_amplitude.py.
EARLY_K = 24
FARNEBACK_KW = dict(
    flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)


def _load_frames(path: pathlib.Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames, axis=0)


def _per_frame_flow(frames_rgb: np.ndarray) -> np.ndarray:
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
    out = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, **FARNEBACK_KW)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        out.append(float(mag.mean()))
    return np.asarray(out, dtype=np.float32)


def _metrics_from_flow(per_frame: np.ndarray, H: int, W: int) -> Dict:
    n = per_frame.shape[0]
    k = min(EARLY_K, n - 1) if n > 1 else n
    early = float(per_frame[:k].mean()) if k > 0 else 0.0
    late = float(per_frame[k:].mean()) if n > k else 0.0
    emr = early / late if late > 1e-6 else 0.0
    tau = 3.0 * min(H, W) / 256.0
    above = np.where(per_frame > tau)[0]
    ttfm = int(above[0]) + 1 if above.size else n
    return {
        "mean_flow_all": float(per_frame.mean()),
        "mean_flow_early": early,
        "mean_flow_late": late,
        "emr": emr,
        "ttfm": ttfm,
        "tau": tau,
        "per_frame_flow": per_frame.tolist(),
    }


def _process_condition(name: str, directory: pathlib.Path) -> Dict:
    mp4s = sorted(directory.glob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"no MP4s in {directory}")

    per_clip = []
    all_per_frame: List[np.ndarray] = []
    for p in mp4s:
        frames = _load_frames(p)
        H, W = frames.shape[1], frames.shape[2]
        per_frame = _per_frame_flow(frames)
        m = _metrics_from_flow(per_frame, H, W)
        m["id"] = p.stem
        per_clip.append(m)
        all_per_frame.append(per_frame)

    # Pad per-frame arrays to the shortest clip so std is well-defined.
    min_len = min(x.shape[0] for x in all_per_frame)
    stacked = np.stack([x[:min_len] for x in all_per_frame], axis=0)
    mean_per_frame = stacked.mean(axis=0).tolist()
    std_per_frame = stacked.std(axis=0).tolist()

    emrs = np.array([c["emr"] for c in per_clip], dtype=np.float32)
    ttfms = np.array([c["ttfm"] for c in per_clip], dtype=np.float32)
    amps = np.array([c["mean_flow_all"] for c in per_clip], dtype=np.float32)
    early_abs = np.array([c["mean_flow_early"] for c in per_clip], dtype=np.float32)
    late_abs = np.array([c["mean_flow_late"] for c in per_clip], dtype=np.float32)

    return {
        "per_clip": per_clip,
        "aggregate": {
            "n": len(per_clip),
            "emr_mean": float(emrs.mean()),
            "emr_std": float(emrs.std(ddof=1)) if emrs.size > 1 else 0.0,
            "ttfm_mean": float(ttfms.mean()),
            "ttfm_std": float(ttfms.std(ddof=1)) if ttfms.size > 1 else 0.0,
            "motion_amp_mean": float(amps.mean()),
            "motion_amp_std": float(amps.std(ddof=1)) if amps.size > 1 else 0.0,
            "early_motion_mean": float(early_abs.mean()),
            "early_motion_std": float(early_abs.std(ddof=1)) if early_abs.size > 1 else 0.0,
            "late_motion_mean": float(late_abs.mean()),
            "late_motion_std": float(late_abs.std(ddof=1)) if late_abs.size > 1 else 0.0,
            "mean_per_frame": mean_per_frame,
            "std_per_frame": std_per_frame,
        },
    }


def _paired_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean_diff, t_stat, p_two_sided). No scipy dependency."""
    d = a - b
    n = d.shape[0]
    if n < 2:
        return float(d.mean()) if n else 0.0, 0.0, 1.0
    m = float(d.mean())
    s = float(d.std(ddof=1))
    if s < 1e-12:
        return m, math.inf if m != 0 else 0.0, 0.0 if m != 0 else 1.0
    t = m / (s / math.sqrt(n))
    # Normal approximation for p; t-statistic is always accurate.
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
    return m, float(t), float(p)


def _paired_comparisons(conditions: Dict[str, Dict]) -> Dict:
    if "t2v" not in conditions:
        return {}
    keys = [k for k in conditions if k != "t2v"]
    t2v_clips = {c["id"]: c for c in conditions["t2v"]["per_clip"]}
    out = {}
    for k in keys:
        k_clips = {c["id"]: c for c in conditions[k]["per_clip"]}
        common = sorted(set(t2v_clips) & set(k_clips))
        if not common:
            continue
        a_emr = np.array([k_clips[i]["emr"] for i in common])
        b_emr = np.array([t2v_clips[i]["emr"] for i in common])
        a_ttfm = np.array([k_clips[i]["ttfm"] for i in common])
        b_ttfm = np.array([t2v_clips[i]["ttfm"] for i in common])
        a_early = np.array([k_clips[i]["mean_flow_early"] for i in common])
        b_early = np.array([t2v_clips[i]["mean_flow_early"] for i in common])
        a_late = np.array([k_clips[i]["mean_flow_late"] for i in common])
        b_late = np.array([t2v_clips[i]["mean_flow_late"] for i in common])
        m, t, p = _paired_t(a_emr, b_emr)
        m2, t2, p2 = _paired_t(a_ttfm, b_ttfm)
        m3, t3, p3 = _paired_t(a_early, b_early)
        m4, t4, p4 = _paired_t(a_late, b_late)
        out[f"{k}_vs_t2v"] = {
            "n": len(common),
            "emr_paired_delta_mean": m, "emr_t_stat": t, "emr_p_value_two_sided": p,
            "ttfm_paired_delta_mean": m2, "ttfm_t_stat": t2, "ttfm_p_value_two_sided": p2,
            "early_motion_paired_delta_mean": m3,
            "early_motion_t_stat": t3,
            "early_motion_p_value_two_sided": p3,
            "late_motion_paired_delta_mean": m4,
            "late_motion_t_stat": t4,
            "late_motion_p_value_two_sided": p4,
        }
    return out


def _plot_time_series(conditions: Dict[str, Dict], figure_path: pathlib.Path,
                      fps: int = 16):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    for name, data in conditions.items():
        agg = data["aggregate"]
        mean = np.array(agg["mean_per_frame"], dtype=np.float32)
        std = np.array(agg["std_per_frame"], dtype=np.float32)
        t_axis = np.arange(1, mean.shape[0] + 1) / fps
        (line,) = ax.plot(t_axis, mean, label=name, linewidth=2)
        ax.fill_between(t_axis, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.axvline(EARLY_K / fps, color="gray", linestyle="--", linewidth=1,
               label=f"K={EARLY_K} frame cutoff")
    ax.set_xlabel("Time since first frame (s)")
    ax.set_ylabel("Mean optical-flow magnitude (px/frame)")
    ax.set_title("Per-frame motion magnitude across conditions\n"
                 "mean ± 1 std across the diagnostic suite")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help="pairs of name=dir (e.g. t2v=outputs/diagnostic/t2v)",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--figure", required=True, type=pathlib.Path)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    conds: Dict[str, Dict] = {}
    for spec in args.conditions:
        if "=" not in spec:
            raise SystemExit(f"bad --conditions spec: {spec}")
        name, path = spec.split("=", 1)
        conds[name] = _process_condition(name, pathlib.Path(path))

    result = {
        "conditions": conds,
        "paired": _paired_comparisons(conds),
        "params": {
            "K": EARLY_K,
            "fps": args.fps,
            "farneback": FARNEBACK_KW,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)
    _plot_time_series(conds, args.figure, fps=args.fps)

    print("=" * 60)
    for name, data in conds.items():
        a = data["aggregate"]
        print(f"[{name}] n={a['n']}  "
              f"EMR={a['emr_mean']:.3f}±{a['emr_std']:.3f}  "
              f"TTFM={a['ttfm_mean']:.2f}±{a['ttfm_std']:.2f}  "
              f"MotionAmp={a['motion_amp_mean']:.3f}  "
              f"Early={a['early_motion_mean']:.3f}  "
              f"Late={a['late_motion_mean']:.3f}")
    for k, v in result["paired"].items():
        print(f"[paired {k}] ΔEMR={v['emr_paired_delta_mean']:+.3f}  "
              f"t={v['emr_t_stat']:+.2f}  p={v['emr_p_value_two_sided']:.4f}")
        print(f"             ΔEarly={v['early_motion_paired_delta_mean']:+.3f}  "
              f"t={v['early_motion_t_stat']:+.2f}  p={v['early_motion_p_value_two_sided']:.4f}")
    print(f"\nwrote {args.output}")
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
