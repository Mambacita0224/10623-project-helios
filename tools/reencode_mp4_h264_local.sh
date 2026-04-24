#!/usr/bin/env bash
# Re-encode OpenCV "mp4v" (MPEG-4 Part 2) MP4s to H.264 for macOS Preview / QuickTime.
# Requires: brew install ffmpeg   (or conda-forge ffmpeg)
#
# Usage (from 10623-project-helios):
#   bash tools/reencode_mp4_h264_local.sh outputs/mixkit_qc30
#   bash tools/reencode_mp4_h264_local.sh outputs/mixkit_qc30 outputs/mixkit_qc30_h264
set -euo pipefail
INDIR="${1:-outputs/mixkit_qc30}"
OUTDIR="${2:-${INDIR}_h264}"
if ! command -v ffmpeg &>/dev/null; then
  echo "Install ffmpeg first:  brew install ffmpeg" >&2
  exit 1
fi
mkdir -p "$OUTDIR"
shopt -s nullglob
FILES=("$INDIR"/*.mp4)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "No .mp4 files in $INDIR" >&2
  exit 1
fi
n=0
for f in "${FILES[@]}"; do
  b=$(basename "$f")
  echo "-> $b"
  ffmpeg -y -i "$f" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -an "$OUTDIR/$b" </dev/null
  n=$((n + 1))
done
echo "Done: $n files -> $OUTDIR"
