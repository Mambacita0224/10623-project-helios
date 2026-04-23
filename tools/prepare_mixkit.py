"""Build the curated Mixkit-Src training manifest for Helios I2V fine-tuning.

Pipeline on a Modal volume:

    1. Download `FastVideo/Mixkit-Src` into `/vol/mixkit_src` (first time).
    2. Walk the motion-oriented category whitelist below.
    3. Extract a centered 49-frame window at 16 fps and 384x640 per clip.
    4. Compute Farneback flow; derive EMR = mean_flow_early / mean_flow_late.
    5. Keep clips with `mean_flow_early >= EARLY_FLOW_MIN` and
       `EMR in [EMR_LOW, EMR_HIGH]`.
    6. Write clip MP4, first-frame PNG, and a manifest line per kept clip.

`manifest.jsonl` is pulled back locally and committed at
`data/train/mixkit_curated.jsonl`; the clips stay on the Modal volume.

Run:

    modal run tools/prepare_mixkit.py::download_source
    modal run tools/prepare_mixkit.py::build_manifest
    modal run tools/prepare_mixkit.py::pull_manifest
"""

from __future__ import annotations

import io
import json
import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_NAME = "helios-mixkit-prep"
VOLUME_NAME = "helios-mixkit"
MOUNT_PATH = "/vol"

# ---- Pipeline constants -----------------------------------------------------

TARGET_FPS = 16
TARGET_FRAMES = 49
TARGET_H = 384
TARGET_W = 640
EARLY_K = 24

# Filtering band (see docs/metrics.md §2.1 for rationale).
EARLY_FLOW_MIN = 3.0
EMR_LOW = 0.7
EMR_HIGH = 1.3

# Immediate-motion category whitelist (Mixkit-Src folder names).
CATEGORIES_KEEP = {
    "Dance", "Sport", "Car", "Cars", "Drive", "Traffic", "Dogs", "Cats",
    "Birds", "Fish", "Shark", "Wildlife", "Motorcycle", "Motocycle",
    "Bicycle", "rain", "fire", "smoke", "sea", "clouds", "Reptiles",
    "Trains", "Truck", "Taxi",
}
# Categories explicitly skipped (for documentation; everything not in KEEP is skipped).
CATEGORIES_SKIP = {"House", "night", "Fashion", "Business", "City", "Music"}

HF_DATASET = "FastVideo/Mixkit-Src"

# ---- Modal image and app ----------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy<2.0.0",
        "huggingface-hub==0.28.1",
        "tqdm",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---- Helpers (executed remotely) --------------------------------------------

def _farneback_flow(prev_gray, cur_gray):
    import cv2

    return cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )


def _extract_clip(src_path: str):
    """Return (frames_rgb: np.ndarray[N,H,W,3] uint8, ok: bool)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return np.empty((0,)), False

    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < int(TARGET_FRAMES * src_fps / TARGET_FPS):
        cap.release()
        return np.empty((0,)), False

    # Stride in source frames that gives us TARGET_FPS from src_fps.
    stride = max(1, round(src_fps / TARGET_FPS))
    needed_src = TARGET_FRAMES * stride
    start = max(0, (total - needed_src) // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    read = 0
    while len(frames) < TARGET_FRAMES and read < needed_src + stride:
        ok, frame = cap.read()
        if not ok:
            break
        if read % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        read += 1
    cap.release()

    if len(frames) < TARGET_FRAMES:
        return np.empty((0,)), False
    return np.stack(frames, axis=0), True


def _compute_flow_stats(frames_rgb):
    """Return dict with mean_flow_all/early/late, emr, ttfm, per_frame."""
    import cv2
    import numpy as np

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
    per_frame = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = _farneback_flow(a, b)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        per_frame.append(float(mag.mean()))
    per_frame = np.asarray(per_frame, dtype=np.float32)

    n_pairs = per_frame.shape[0]
    k = min(EARLY_K, n_pairs - 1)
    early = float(per_frame[:k].mean())
    late = float(per_frame[k:].mean())
    emr = early / late if late > 1e-6 else 0.0

    tau = 3.0 * min(TARGET_H, TARGET_W) / 256.0
    above = np.where(per_frame > tau)[0]
    ttfm = int(above[0]) + 1 if above.size else n_pairs

    return {
        "mean_flow_all": float(per_frame.mean()),
        "mean_flow_early": early,
        "mean_flow_late": late,
        "emr": emr,
        "ttfm": ttfm,
        "per_frame_flow": per_frame.tolist(),
    }


def _save_clip_mp4(frames_rgb, out_path: str):
    import cv2

    h, w = frames_rgb.shape[1], frames_rgb.shape[2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (w, h),
    )
    for f in frames_rgb:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _caption_from_filename(fname: str, category: str) -> str:
    stem = pathlib.Path(fname).stem
    # Mixkit stems look like `running-girl-in-park-123`. Humanize.
    words = [w for w in stem.replace("_", "-").split("-") if not w.isdigit()]
    body = " ".join(words).strip().lower() or category.lower()
    return f"{body}, dynamic motion from the first frame"


# ---- Modal functions --------------------------------------------------------

@app.function(
    volumes={MOUNT_PATH: volume},
    timeout=60 * 60 * 3,
    cpu=8,
    memory=16 * 1024,
)
def download_source():
    """One-shot: download FastVideo/Mixkit-Src into the volume."""
    import subprocess

    dst = pathlib.Path(MOUNT_PATH) / "mixkit_src"
    if dst.exists() and any(dst.iterdir()):
        print(f"[skip] {dst} already populated")
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_DATASET} into {dst} ...")
    subprocess.check_call([
        "huggingface-cli", "download", HF_DATASET,
        "--repo-type", "dataset",
        "--local-dir", str(dst),
        "--local-dir-use-symlinks", "False",
    ])
    volume.commit()
    print(f"[ok] download complete at {dst}")
    return str(dst)


@app.function(
    volumes={MOUNT_PATH: volume},
    timeout=60 * 60 * 6,
    cpu=16,
    memory=32 * 1024,
)
def build_manifest(max_per_category: int = 60):
    """Walk the whitelisted categories, filter, and write the manifest."""
    import cv2  # noqa: F401  (ensure imports work in the runtime)

    src_root = pathlib.Path(MOUNT_PATH) / "mixkit_src"
    out_root = pathlib.Path(MOUNT_PATH) / "mixkit_curated"
    clips_dir = out_root / "clips"
    frames_dir = out_root / "first_frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise RuntimeError(
            f"source not found at {src_root}; run `download_source` first"
        )

    cat_dirs = sorted(
        [d for d in src_root.iterdir() if d.is_dir() and d.name in CATEGORIES_KEEP]
    )
    print(f"found {len(cat_dirs)} whitelisted categories: {[d.name for d in cat_dirs]}")

    manifest_path = out_root / "manifest.jsonl"
    stats = {
        "attempted": 0, "read_failed": 0,
        "filtered_early_flow": 0, "filtered_emr": 0, "kept": 0,
    }
    next_id = 0

    with manifest_path.open("w") as manifest:
        for cat_dir in cat_dirs:
            cat_kept = 0
            src_files = sorted(p for p in cat_dir.rglob("*.mp4") if p.is_file())
            print(f"\n[{cat_dir.name}] {len(src_files)} source clips")
            for src in src_files:
                if cat_kept >= max_per_category:
                    break
                stats["attempted"] += 1
                frames, ok = _extract_clip(str(src))
                if not ok:
                    stats["read_failed"] += 1
                    continue
                flow = _compute_flow_stats(frames)
                if flow["mean_flow_early"] < EARLY_FLOW_MIN:
                    stats["filtered_early_flow"] += 1
                    continue
                if not (EMR_LOW <= flow["emr"] <= EMR_HIGH):
                    stats["filtered_emr"] += 1
                    continue

                clip_id = f"{next_id:05d}"
                next_id += 1
                clip_path = clips_dir / f"{clip_id}.mp4"
                frame_path = frames_dir / f"{clip_id}.png"
                _save_clip_mp4(frames, str(clip_path))
                import cv2
                cv2.imwrite(str(frame_path),
                            cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

                record = {
                    "id": clip_id,
                    "category": cat_dir.name,
                    "source_path": str(src.relative_to(src_root)),
                    "prompt": _caption_from_filename(src.name, cat_dir.name),
                    "clip_path": str(clip_path.relative_to(out_root)),
                    "first_frame_path": str(frame_path.relative_to(out_root)),
                    "mean_flow_all": flow["mean_flow_all"],
                    "mean_flow_early": flow["mean_flow_early"],
                    "mean_flow_late": flow["mean_flow_late"],
                    "emr": flow["emr"],
                    "ttfm": flow["ttfm"],
                }
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                stats["kept"] += 1
                cat_kept += 1
                if cat_kept % 10 == 0:
                    print(f"  [{cat_dir.name}] kept {cat_kept} so far")

            print(f"  [{cat_dir.name}] total kept {cat_kept}")

    volume.commit()
    print("\n" + "=" * 50)
    print(f"attempted:           {stats['attempted']}")
    print(f"  read_failed:       {stats['read_failed']}")
    print(f"  filtered_early:    {stats['filtered_early_flow']}")
    print(f"  filtered_emr:      {stats['filtered_emr']}")
    print(f"  kept:              {stats['kept']}")
    print(f"manifest: {manifest_path}")
    return stats


@app.function(volumes={MOUNT_PATH: volume})
def _read_manifest_bytes() -> bytes:
    p = pathlib.Path(MOUNT_PATH) / "mixkit_curated" / "manifest.jsonl"
    return p.read_bytes()


@app.local_entrypoint()
def pull_manifest(out_path: str = "data/train/mixkit_curated.jsonl"):
    """Copy the remote manifest.jsonl to the local repo for committing."""
    dst = REPO_ROOT / out_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_manifest_bytes.remote()
    dst.write_bytes(payload)
    n = sum(1 for _ in io.BytesIO(payload))
    print(f"[ok] wrote {n} manifest lines to {dst}")
