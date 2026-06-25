#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DEFAULT_DATASET_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
DEFAULT_OUTPUT_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/runs"

if [[ -z "${CGH_DATASET_ROOT:-}" && -d "$DEFAULT_DATASET_ROOT" ]]; then
  export CGH_DATASET_ROOT="$DEFAULT_DATASET_ROOT"
fi
if [[ -z "${CGH_OUTPUT_ROOT:-}" ]]; then
  export CGH_OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
fi

if [[ -z "${CGH_DATASET_ROOT:-}" || ! -d "$CGH_DATASET_ROOT" ]]; then
  echo "ERROR: CGH_DATASET_ROOT is not set or does not exist." >&2
  echo "Run first: bash training/scripts/setup_sutd_data.sh" >&2
  exit 2
fi

echo "Using CGH_DATASET_ROOT=$CGH_DATASET_ROOT"
echo "Using CGH_OUTPUT_ROOT=$CGH_OUTPUT_ROOT"

if command -v conda >/dev/null 2>&1 && [[ -n "${CGH_CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CGH_CONDA_ENV"
fi

export PATH="$HOME/.local/bin:$PATH"
if [[ "${CGH_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --user -r requirements.txt
else
  echo "Skipping pip install because CGH_SKIP_PIP_INSTALL=1"
fi
python training/run_pipeline.py --config training/configs/sutd.yaml --stage infer --visualize 2>&1 | tee "sutd_inference.log"
python training/run_pipeline.py --config training/configs/sutd.yaml --stage morphology 2>&1 | tee -a "sutd_inference.log"
