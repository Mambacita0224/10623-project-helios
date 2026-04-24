#!/usr/bin/env bash
set -euo pipefail
VOLUME="helios-mixkit"
DEST="${1:-/Users/zengyuhang/Desktop/academic/2025-2026 spring/10623 GenAI/project/10623-project-helios/outputs/mixkit_qc30}"
mkdir -p "$DEST"
PATHS_FILE="/Users/zengyuhang/Desktop/academic/2025-2026 spring/10623 GenAI/project/10623-project-helios/data/train/mixkit_qc30_volume_paths.txt"
while IFS= read -r p; do
  [ -n "$p" ] || continue
  base=$(basename "$p")
  echo "-> $p"
  modal volume get "$VOLUME" "$p" "$DEST/$base" --force
done < "$PATHS_FILE"
echo "Done. Open: $DEST"
