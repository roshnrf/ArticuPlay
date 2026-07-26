#!/bin/bash
# Pulls exactly what Phase 1 needs from the public UltraSuite rsync mirror:
# UXSSD audio (wav-only, skips ultrasound), word-boundary labels, and the
# real clinician pronunciation-score CSV. No auth needed, no ultrasound data.
set -e

DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR/core-uxssd" "$DATA_DIR/labels"

echo "--- pulling UXSSD audio (wav-only) ---"
rsync -av --include='*/' --include='*.wav' --exclude='*' \
  ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxssd/ "$DATA_DIR/core-uxssd/"

echo "--- pulling word-boundary labels (.lab format) ---"
rsync -av \
  ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxssd/reference_labels/word-labels/lab/ \
  "$DATA_DIR/labels/word-labels/"

echo "--- pulling pronunciation scores CSV ---"
rsync -av \
  ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/labels-uxtd-uxssd-upx/uxssd/pronunciation_scores/uxssd-pronunciation-scores.csv \
  "$DATA_DIR/uxssd-pronunciation-scores.csv"

echo "--- done ---"
echo "wav files: $(find "$DATA_DIR/core-uxssd" -name '*.wav' | wc -l)"
echo "label files: $(find "$DATA_DIR/labels/word-labels" -name '*.lab' | wc -l)"
echo "csv rows: $(($(wc -l < "$DATA_DIR/uxssd-pronunciation-scores.csv") - 1))"
