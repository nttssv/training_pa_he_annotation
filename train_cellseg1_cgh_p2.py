#!/usr/bin/env python3
"""Train CellSeg1 on the exported CGH PA P2 ground-truth tiles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="/Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing/qupath_work/cellseg1_repo",
        help="Path to the cloned Nuisal/cellseg1 repository.",
    )
    parser.add_argument(
        "--config",
        default="/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/cgh_p2_cellseg1_config.yaml",
        help="Path to the CellSeg1 YAML config.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"CellSeg1 repo not found: {repo}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    sys.path.insert(0, str(repo))
    from cellseg1_train import main as train_main

    train_main(config)


if __name__ == "__main__":
    main()
