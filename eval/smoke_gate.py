#!/usr/bin/env python3
"""Run smoke quality gate and decide whether to proceed to full training.

This wrapper reuses eval/compute_emr_ttfm.py to keep metric definitions identical
with your existing diagnostic pipeline, then applies a PASS/FAIL gate.

Default expectation:
- baseline videos in outputs/diagnostic/mixkit/i2v
- smoke LoRA videos in outputs/diagnostic/mixkit/i2v_lora

Exit codes:
- 0: PASS (safe to move to full)
- 2: FAIL (tune and rerun smoke)
- 1: runtime/config error
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_metrics(
    repo_root: Path,
    baseline_dir: Path,
    candidate_dir: Path,
    t2v_dir: Path | None,
    metrics_json: Path,
    metrics_figure: Path,
    fps: int,
) -> None:
    cmd = [
        sys.executable,
        "eval/compute_emr_ttfm.py",
        "--conditions",
        f"baseline={baseline_dir}",
        f"candidate={candidate_dir}",
    ]
    if t2v_dir is not None:
        cmd.append(f"t2v={t2v_dir}")
    cmd.extend([
        "--output",
        str(metrics_json),
        "--figure",
        str(metrics_figure),
        "--fps",
        str(fps),
    ])
    subprocess.run(cmd, check=True, cwd=repo_root)


def _check(condition: bool, title: str, detail: str) -> dict:
    return {
        "name": title,
        "passed": bool(condition),
        "detail": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/diagnostic/mixkit/i2v"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("outputs/diagnostic/mixkit/i2v_lora"))
    parser.add_argument("--t2v-dir", type=Path, default=Path("outputs/diagnostic/t2v"))
    parser.add_argument("--fps", type=int, default=24)

    parser.add_argument("--min-early-abs", type=float, default=0.18)
    parser.add_argument("--min-late-abs", type=float, default=0.45)
    parser.add_argument("--min-ttfm", type=float, default=90.0)

    parser.add_argument("--min-early-ratio", type=float, default=0.70)
    parser.add_argument("--min-late-ratio", type=float, default=0.70)
    parser.add_argument("--min-motion-amp-ratio", type=float, default=0.70)

    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("outputs/diagnostic/smoke_gate_metrics.json"),
    )
    parser.add_argument(
        "--metrics-figure",
        type=Path,
        default=Path("outputs/diagnostic/smoke_gate_figure.png"),
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        default=Path("outputs/diagnostic/smoke_gate_decision.json"),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    baseline_dir = (repo_root / args.baseline_dir).resolve()
    candidate_dir = (repo_root / args.candidate_dir).resolve()
    t2v_dir = (repo_root / args.t2v_dir).resolve()
    metrics_json = (repo_root / args.metrics_json).resolve()
    metrics_figure = (repo_root / args.metrics_figure).resolve()
    decision_json = (repo_root / args.decision_json).resolve()

    if not baseline_dir.is_dir():
        raise SystemExit(f"baseline_dir not found: {baseline_dir}")
    if not candidate_dir.is_dir():
        raise SystemExit(f"candidate_dir not found: {candidate_dir}")

    use_t2v_dir = t2v_dir if t2v_dir.is_dir() else None

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_figure.parent.mkdir(parents=True, exist_ok=True)
    decision_json.parent.mkdir(parents=True, exist_ok=True)

    _run_metrics(
        repo_root=repo_root,
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        t2v_dir=use_t2v_dir,
        metrics_json=metrics_json,
        metrics_figure=metrics_figure,
        fps=args.fps,
    )

    with metrics_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conds = data["conditions"]
    baseline = conds["baseline"]["aggregate"]
    candidate = conds["candidate"]["aggregate"]

    early_ratio = candidate["early_motion_mean"] / max(baseline["early_motion_mean"], 1e-8)
    late_ratio = candidate["late_motion_mean"] / max(baseline["late_motion_mean"], 1e-8)
    amp_ratio = candidate["motion_amp_mean"] / max(baseline["motion_amp_mean"], 1e-8)

    checks = [
        _check(
            candidate["early_motion_mean"] >= args.min_early_abs,
            "early_motion_abs",
            f"candidate={candidate['early_motion_mean']:.4f}, threshold={args.min_early_abs:.4f}",
        ),
        _check(
            candidate["late_motion_mean"] >= args.min_late_abs,
            "late_motion_abs",
            f"candidate={candidate['late_motion_mean']:.4f}, threshold={args.min_late_abs:.4f}",
        ),
        _check(
            candidate["ttfm_mean"] >= args.min_ttfm,
            "ttfm_abs",
            f"candidate={candidate['ttfm_mean']:.2f}, threshold={args.min_ttfm:.2f}",
        ),
        _check(
            early_ratio >= args.min_early_ratio,
            "early_motion_vs_baseline",
            f"ratio={early_ratio:.3f}, threshold={args.min_early_ratio:.3f}",
        ),
        _check(
            late_ratio >= args.min_late_ratio,
            "late_motion_vs_baseline",
            f"ratio={late_ratio:.3f}, threshold={args.min_late_ratio:.3f}",
        ),
        _check(
            amp_ratio >= args.min_motion_amp_ratio,
            "motion_amp_vs_baseline",
            f"ratio={amp_ratio:.3f}, threshold={args.min_motion_amp_ratio:.3f}",
        ),
    ]

    passed = all(c["passed"] for c in checks)

    decision = {
        "passed": passed,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "thresholds": {
            "min_early_abs": args.min_early_abs,
            "min_late_abs": args.min_late_abs,
            "min_ttfm": args.min_ttfm,
            "min_early_ratio": args.min_early_ratio,
            "min_late_ratio": args.min_late_ratio,
            "min_motion_amp_ratio": args.min_motion_amp_ratio,
        },
        "baseline": {
            "early_motion_mean": baseline["early_motion_mean"],
            "late_motion_mean": baseline["late_motion_mean"],
            "ttfm_mean": baseline["ttfm_mean"],
            "motion_amp_mean": baseline["motion_amp_mean"],
        },
        "candidate": {
            "early_motion_mean": candidate["early_motion_mean"],
            "late_motion_mean": candidate["late_motion_mean"],
            "ttfm_mean": candidate["ttfm_mean"],
            "motion_amp_mean": candidate["motion_amp_mean"],
        },
        "ratios": {
            "early_motion": early_ratio,
            "late_motion": late_ratio,
            "motion_amp": amp_ratio,
        },
        "checks": checks,
        "metrics_json": str(metrics_json),
        "metrics_figure": str(metrics_figure),
    }

    with decision_json.open("w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    print("=" * 68)
    print("Smoke Gate Result:", "PASS" if passed else "FAIL")
    print(f"baseline:  early={baseline['early_motion_mean']:.4f} late={baseline['late_motion_mean']:.4f} ttfm={baseline['ttfm_mean']:.2f} amp={baseline['motion_amp_mean']:.4f}")
    print(f"candidate: early={candidate['early_motion_mean']:.4f} late={candidate['late_motion_mean']:.4f} ttfm={candidate['ttfm_mean']:.2f} amp={candidate['motion_amp_mean']:.4f}")
    print(f"ratios:    early={early_ratio:.3f} late={late_ratio:.3f} amp={amp_ratio:.3f}")
    for c in checks:
        tag = "PASS" if c["passed"] else "FAIL"
        print(f"[{tag}] {c['name']}: {c['detail']}")
    print(f"decision json: {decision_json}")
    print(f"metrics json:  {metrics_json}")
    print(f"figure:        {metrics_figure}")
    print("=" * 68)

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
