#!/bin/bash
# Pulls UXTD (typically-developing children) audio + prompt text, wav-only.
# No mislabeling risk here (unlike UXSSD) — TD kids have no disorder, so the
# prompt text they were asked to say reliably matches what they actually said.
set -e

DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR/core-uxtd"

echo "--- pulling UXTD audio + prompt text (wav-only + txt) ---"
rsync -av --include='*/' --include='*.wav' --include='*.txt' --exclude='*' \
  ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd/ "$DATA_DIR/core-uxtd/"

echo "--- done ---"
echo "wav files: $(find "$DATA_DIR/core-uxtd" -name '*.wav' | wc -l)"
echo "txt files: $(find "$DATA_DIR/core-uxtd" -name '*.txt' | wc -l)"
echo "speakers: $(find "$DATA_DIR/core-uxtd/core" -mindepth 1 -maxdepth 1 -type d | wc -l)"
