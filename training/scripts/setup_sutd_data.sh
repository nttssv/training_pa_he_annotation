#!/usr/bin/env bash
set -euo pipefail

DATA_PARENT="${CGH_DATA_PARENT:-$HOME/Desktop/1.Data}"
DATA_REPO_DIR="$DATA_PARENT/training_pa_he_annotation_full"
DATASET_DIR="$DATA_REPO_DIR/cellseg1_cgh_p2_combined_41_full"
DATA_REPO_URL="${CGH_DATA_REPO_URL:-https://github.com/nttssv/training_pa_he_annotation.git}"
DATA_BRANCH="${CGH_DATA_BRANCH:-codex/add-second-batch-training-data}"

mkdir -p "$DATA_PARENT"

if [[ -d "$DATA_REPO_DIR/.git" ]]; then
  echo "Using existing data repo: $DATA_REPO_DIR"
  git -C "$DATA_REPO_DIR" fetch origin "$DATA_BRANCH"
  git -C "$DATA_REPO_DIR" switch "$DATA_BRANCH"
  git -C "$DATA_REPO_DIR" pull --ff-only origin "$DATA_BRANCH"
else
  echo "Cloning data repo into: $DATA_REPO_DIR"
  git clone --branch "$DATA_BRANCH" --single-branch "$DATA_REPO_URL" "$DATA_REPO_DIR"
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: Expected dataset folder not found: $DATASET_DIR" >&2
  echo "Check that branch '$DATA_BRANCH' contains cellseg1_cgh_p2_combined_41_full/." >&2
  exit 2
fi

echo "Dataset image count:"
find "$DATASET_DIR/train/images" -type f | wc -l
echo "Dataset mask count:"
find "$DATASET_DIR/train/masks" -type f | wc -l

echo
echo "Data ready."
echo "Run these before training:"
echo "export CGH_DATASET_ROOT=\"$DATASET_DIR\""
echo "export CGH_OUTPUT_ROOT=\"$DATA_REPO_DIR/outputs/runs\""
