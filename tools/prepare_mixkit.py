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
    modal run tools/prepare_mixkit.py::smoke_process_pool   # optional: MP + OpenCV
    modal run tools/prepare_mixkit.py::build_manifest
    modal run tools/prepare_mixkit.py::pull_manifest
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import uuid

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_NAME = "helios-mixkit-prep"
VOLUME_NAME = "helios-mixkit"
MOUNT_PATH = "/vol"
# Written only when huggingface-cli snapshot finishes; avoids skipping re-runs after partial 429s.
DOWNLOAD_DONE_MARKER = ".helios_mixkit_download_complete"
# Same secret name as modal/README.md (HF_TOKEN in the container).
_HF_MODAL_SECRET = "huggingface-secret"

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


# ---- Multiprocessing workers (top-level, picklable) -----------------------------


def _smoke_one_clip(src_path: str) -> str:
    """Single-file probe for `smoke_process_pool` (OpenCV + flow)."""
    frames, ok = _extract_clip(src_path)
    if not ok:
        return f"read_failed:{pathlib.Path(src_path).name}"
    flow = _compute_flow_stats(frames)
    return (
        f"ok file={pathlib.Path(src_path).name} early={flow['mean_flow_early']:.3f} "
        f"emr={flow['emr']:.3f}"
    )


def _process_src_file(
    item: tuple[str, str, str, str],
) -> dict:
    """One source MP4: decode, flow filter, write staging clip+PNG if kept.

    item: (abs_src_path, category_name, src_root, staging_root)
    """
    import cv2

    src_path, category_name, src_root, staging_root = item
    p = pathlib.Path(src_path)
    frames, ok = _extract_clip(str(p))
    if not ok:
        return {
            "src_path": str(p),
            "category": category_name,
            "status": "read_failed",
        }
    flow = _compute_flow_stats(frames)
    if flow["mean_flow_early"] < EARLY_FLOW_MIN:
        return {
            "src_path": str(p),
            "category": category_name,
            "status": "filtered_early_flow",
        }
    if not (EMR_LOW <= flow["emr"] <= EMR_HIGH):
        return {
            "src_path": str(p),
            "category": category_name,
            "status": "filtered_emr",
        }

    tok = uuid.uuid4().hex
    staging = pathlib.Path(staging_root)
    staging.mkdir(parents=True, exist_ok=True)
    mp4_p = str(staging / f"{tok}.mp4")
    png_p = str(staging / f"{tok}.png")
    _save_clip_mp4(frames, mp4_p)
    cv2.imwrite(png_p, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    return {
        "src_path": str(p),
        "category": category_name,
        "status": "kept",
        "staging_mp4": mp4_p,
        "staging_png": png_p,
        "source_path": str(p.relative_to(pathlib.Path(src_root))),
        "prompt": _caption_from_filename(p.name, category_name),
        "mean_flow_all": flow["mean_flow_all"],
        "mean_flow_early": flow["mean_flow_early"],
        "mean_flow_late": flow["mean_flow_late"],
        "emr": flow["emr"],
        "ttfm": flow["ttfm"],
    }


# ---- Modal functions --------------------------------------------------------

@app.function(
    volumes={MOUNT_PATH: volume},
    timeout=60 * 60 * 3,
    cpu=8,
    memory=16 * 1024,
    secrets=[modal.Secret.from_name(_HF_MODAL_SECRET)],
)
def download_source():
    """One-shot: download FastVideo/Mixkit-Src into the volume."""
    import subprocess

    dst = pathlib.Path(MOUNT_PATH) / "mixkit_src"
    done = dst / DOWNLOAD_DONE_MARKER
    if done.exists():
        print(f"[skip] marker exists: {done}")
        return str(dst)
    if not os.environ.get("HF_TOKEN"):
        print(
            "[warn] HF_TOKEN is not set. Create: modal secret create huggingface-secret "
            "HF_TOKEN=hf_... — unauthenticated downloads often hit HTTP 429."
        )
    dst.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HF_DATASET} into {dst} ...")
    subprocess.check_call(
        [
            "huggingface-cli",
            "download",
            HF_DATASET,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dst),
            "--local-dir-use-symlinks",
            "False",
        ],
        env={**os.environ},
    )
    done.write_text("ok\n")
    volume.commit()
    print(f"[ok] download complete at {dst}")
    return str(dst)


def _unlink_quiet(p: str) -> None:
    try:
        pathlib.Path(p).unlink(missing_ok=True)
    except OSError:
        pass


@app.function(
    volumes={MOUNT_PATH: volume},
    timeout=60 * 20,
    cpu=8,
    memory=8 * 1024,
)
def smoke_process_pool(
    num_files: int = 12,
    num_workers: int = 4,
):
    """Run `_smoke_one_clip` in a `ProcessPoolExecutor` to verify OpenCV+flow+MP.

    Call after `download_source` so `mixkit_src` is populated.
    """
    from concurrent.futures import ProcessPoolExecutor

    src_root = pathlib.Path(MOUNT_PATH) / "mixkit_src"
    if not src_root.is_dir() or not any(src_root.iterdir()):
        raise RuntimeError(
            f"Run download_source first; missing or empty {src_root}"
        )

    found: list[pathlib.Path] = []
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if d.name not in CATEGORIES_KEEP:
            continue
        cands = sorted(x for x in d.rglob("*.mp4") if x.is_file())[:num_files]
        if cands:
            found = cands[:num_files]
            break
    if not found:
        cands = sorted(p for p in src_root.rglob("*.mp4") if p.is_file())[:num_files]
        found = cands
    if not found:
        raise RuntimeError("No .mp4 under mixkit_src")

    n_workers = max(1, min(num_workers, len(found), os.cpu_count() or 8))
    print(
        f"[smoke_process_pool] files={len(found)} workers={n_workers} "
        f"cpu_count={os.cpu_count()}"
    )
    paths = [str(p) for p in found]
    mctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mctx) as ex:
        lines = list(ex.map(_smoke_one_clip, paths))
    for line in lines:
        print(line)
    ok = sum(1 for l in lines if l.startswith("ok"))
    return {"n": len(lines), "ok_count": ok, "lines": lines}


@app.function(
    volumes={MOUNT_PATH: volume},
    timeout=60 * 60 * 6,
    cpu=16,
    memory=32 * 1024,
)
def build_manifest(
    max_per_category: int = 60,
    num_workers: int | None = None,
    batch_size: int | None = None,
):
    """Walk the whitelisted categories, filter, and write the manifest (parallel)."""
    from concurrent.futures import ProcessPoolExecutor

    n_workers = num_workers or min(16, max(1, os.cpu_count() or 8))
    bs = batch_size or max(8, min(64, n_workers * 4))
    print(f"[build_manifest] num_workers={n_workers} batch_size={bs}")

    src_root = pathlib.Path(MOUNT_PATH) / "mixkit_src"
    out_root = pathlib.Path(MOUNT_PATH) / "mixkit_curated"
    clips_dir = out_root / "clips"
    frames_dir = out_root / "first_frames"
    staging_root = out_root / "_staging"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

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
        "attempted": 0,
        "read_failed": 0,
        "filtered_early_flow": 0,
        "filtered_emr": 0,
        "kept": 0,
    }
    next_id = 0
    src_s = str(src_root)
    st_s = str(staging_root)

    with manifest_path.open("w") as manifest:
        for cat_dir in cat_dirs:
            src_files = sorted(p for p in cat_dir.rglob("*.mp4") if p.is_file())
            print(f"\n[{cat_dir.name}] {len(src_files)} source clips")
            cat_kept = 0
            i = 0
            while cat_kept < max_per_category and i < len(src_files):
                nbatch = min(bs, len(src_files) - i)
                batch = src_files[i : i + nbatch]
                i += nbatch
                stats["attempted"] += len(batch)
                payloads = [
                    (str(s), cat_dir.name, src_s, st_s) for s in batch
                ]
                mctx = mp.get_context("fork")
                with ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=mctx
                ) as ex:
                    results = list(ex.map(_process_src_file, payloads))
                for res in results:
                    status = res["status"]
                    if status == "read_failed":
                        stats["read_failed"] += 1
                    elif status == "filtered_early_flow":
                        stats["filtered_early_flow"] += 1
                    elif status == "filtered_emr":
                        stats["filtered_emr"] += 1
                    elif status == "kept":
                        if cat_kept >= max_per_category:
                            _unlink_quiet(res["staging_mp4"])
                            _unlink_quiet(res["staging_png"])
                            continue
                        clip_id = f"{next_id:05d}"
                        next_id += 1
                        clip_path = clips_dir / f"{clip_id}.mp4"
                        frame_path = frames_dir / f"{clip_id}.png"
                        shutil.move(res["staging_mp4"], clip_path)
                        shutil.move(res["staging_png"], frame_path)
                        record = {
                            "id": clip_id,
                            "category": res["category"],
                            "source_path": res["source_path"],
                            "prompt": res["prompt"],
                            "clip_path": str(clip_path.relative_to(out_root)),
                            "first_frame_path": str(frame_path.relative_to(out_root)),
                            "mean_flow_all": res["mean_flow_all"],
                            "mean_flow_early": res["mean_flow_early"],
                            "mean_flow_late": res["mean_flow_late"],
                            "emr": res["emr"],
                            "ttfm": res["ttfm"],
                        }
                        manifest.write(json.dumps(record) + "\n")
                        manifest.flush()
                        stats["kept"] += 1
                        cat_kept += 1
                        if cat_kept % 10 == 0:
                            print(
                                f"  [{cat_dir.name}] kept {cat_kept} so far"
                            )
                    else:  # pragma: no cover
                        pass
                if cat_kept >= max_per_category:
                    break

            print(f"  [{cat_dir.name}] total kept {cat_kept}")

    # Best-effort cleanup of empty staging
    try:
        for p in staging_root.iterdir():
            p.unlink()
        staging_root.rmdir()
    except OSError:
        pass

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
