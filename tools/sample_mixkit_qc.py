#!/usr/bin/env python3
"""Sample N clips from data/train/mixkit_curated.jsonl for manual QC (Modal volume paths).

Generates a JSON manifest, one path per line for Modal, and a shell script to
`modal volume get` them into a local folder (e.g. outputs/mixkit_qc30/).

Usage:
  python tools/sample_mixkit_qc.py -n 30 --seed 42
  bash data/train/mixkit_qc30_download.sh
  open outputs/mixkit_qc30   # play in VLC; Preview may not like OpenCV mp4v

If macOS Preview won't play: older builds used OpenCV mp4v. Re-encode with
  bash tools/reencode_mp4_h264_local.sh outputs/mixkit_qc30
New `build_manifest` runs use ffmpeg H.264 (Preview-friendly).
"""
from __future__ import annotations

import argparse
import json
import random
import stat
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_JSONL = REPO / "data/train/mixkit_curated.jsonl"
VOLUME_NAME = "helios-mixkit"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("Usage:")[0].strip())
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Curated manifest jsonl")
    p.add_argument("-n", type=int, default=30, help="Number of clips to sample")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible sample)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/train",
        help="Where to write manifest + download script (tracked path)",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=REPO / "outputs/mixkit_qc30",
        help="Path written into the download script (default under outputs/, gitignored videos)",
    )
    args = p.parse_args()

    lines = [json.loads(s) for s in args.jsonl.read_text().splitlines() if s.strip()]
    if not lines:
        raise SystemExit(f"no records in {args.jsonl}")
    n = min(args.n, len(lines))
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(len(lines)), n))
    sample = [lines[i] for i in picked]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mixkit_qc{n}"
    man_path = out_dir / f"{stem}_manifest.json"
    man_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")

    paths: list[str] = []
    for r in sample:
        rel = (r.get("clip_path") or "").replace("\\", "/").lstrip("/")
        if not rel.startswith("mixkit_curated/"):
            rel = f"mixkit_curated/{rel}"
        paths.append(rel)

    paths_path = out_dir / f"{stem}_volume_paths.txt"
    paths_path.write_text("\n".join(paths) + "\n", encoding="utf-8")

    local_dir = args.local_dir
    if not local_dir.is_absolute():
        local_dir = (REPO / local_dir).resolve()

    sh = out_dir / f"{stem}_download.sh"
    sh_content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        VOLUME="{VOLUME_NAME}"
        DEST="${{1:-{local_dir}}}"
        mkdir -p "$DEST"
        PATHS_FILE="{paths_path}"
        while IFS= read -r p; do
          [ -n "$p" ] || continue
          base=$(basename "$p")
          echo "-> $p"
          modal volume get "$VOLUME" "$p" "$DEST/$base" --force
        done < "$PATHS_FILE"
        echo "Done. Open: $DEST"
        """
    )
    sh.write_text(sh_content, encoding="utf-8")
    sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Wrote {man_path} ({n} records)")
    print(f"Wrote {paths_path}")
    print(f"Wrote {sh} — run: bash {sh}")
    print(f"  (downloads into {local_dir} by default)")


if __name__ == "__main__":
    main()