#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if command -v conda >/dev/null 2>&1 && [[ -n "${CGH_CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CGH_CONDA_ENV"
fi

python -m pip install -r requirements.txt
python training/run_pipeline.py --config training/configs/sutd.yaml --stage smoke --visualize 2>&1 | tee "sutd_smoke_test.log"

