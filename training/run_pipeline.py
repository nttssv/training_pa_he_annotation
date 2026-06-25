"""Config-driven adrenal H&E morphology training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import audit_dataset, compute_morphology, evaluate_pipeline, infer_pipeline, match_nucleus_boundary
from training import split_dataset, train_cell_boundary, train_yolo_nuclei, visualize_dataset
from training.utils.io import PipelineError, load_config, write_json
from training.utils.progress import write_error_report, write_run_summary
from training.utils.reproducibility import (
    copy_config_files,
    require_cuda_if_configured,
    run_dirs,
    save_reproducibility_snapshot,
    set_seed,
    timestamped_run_dir,
)
from training.utils.visualization import write_html_report


StageFn = Callable[[dict[str, Any], dict[str, Path]], dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "audit",
            "split",
            "visualize",
            "smoke",
            "train_nuclei",
            "train_boundary",
            "infer",
            "morphology",
            "evaluate",
            "all",
        ],
    )
    parser.add_argument("--visualize", action="store_true", help="Generate dataset and checkpoint visualizations.")
    return parser


def initialize_run(cfg: dict[str, Any]) -> dict[str, Path]:
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    run_dir = timestamped_run_dir(cfg)
    dirs = run_dirs(run_dir)
    copy_config_files(cfg, dirs)
    save_reproducibility_snapshot(cfg, dirs)
    return dirs


def run_stage(stage: str, cfg: dict[str, Any], dirs: dict[str, Path], visualize: bool = False) -> dict[str, Any]:
    if stage == "audit":
        return audit_dataset.run(cfg, dirs)
    if stage == "split":
        return split_dataset.run(cfg, dirs)
    if stage == "visualize":
        return visualize_dataset.run(cfg, dirs)
    if stage == "train_nuclei":
        return train_yolo_nuclei.run(cfg, dirs)
    if stage == "train_boundary":
        if visualize:
            visualize_dataset.run(cfg, dirs)
        return train_cell_boundary.run(cfg, dirs)
    if stage == "infer":
        return infer_pipeline.run(cfg, dirs)
    if stage == "morphology":
        return compute_morphology.run(cfg, dirs)
    if stage == "evaluate":
        return evaluate_pipeline.run(cfg, dirs)
    if stage == "smoke":
        return run_smoke(cfg, dirs, visualize=visualize)
    if stage == "all":
        return run_all(cfg, dirs, visualize=visualize)
    raise PipelineError("config", f"Unsupported stage: {stage}")


def run_smoke(cfg: dict[str, Any], dirs: dict[str, Path], visualize: bool = False) -> dict[str, Any]:
    require_cuda_if_configured(cfg)
    audit = audit_dataset.run(cfg, dirs)
    split = split_dataset.run(cfg, dirs)
    write_run_summary(dirs["logs"] / "run_summary.txt", cfg, split, dirs["run"])
    if visualize:
        visualize_dataset.run(cfg, dirs)
    smoke = train_cell_boundary.smoke(cfg, dirs, visualize=True)
    infer = infer_pipeline.infer(cfg, dirs, limit=int(cfg.get("smoke", {}).get("num_images", 3)))
    matching = match_nucleus_boundary.run(cfg, dirs)
    morphology = compute_morphology.run(cfg, dirs)
    result = {
        "audit": audit,
        "split": split,
        "smoke": smoke,
        "infer": infer,
        "matching": matching,
        "morphology": morphology,
    }
    write_json(dirs["metrics"] / "smoke_pipeline_summary.json", result)
    update_report(cfg, dirs, result)
    return result


def run_all(cfg: dict[str, Any], dirs: dict[str, Path], visualize: bool = False) -> dict[str, Any]:
    require_cuda_if_configured(cfg)
    result: dict[str, Any] = {}
    result["audit"] = audit_dataset.run(cfg, dirs)
    result["split"] = split_dataset.run(cfg, dirs)
    write_run_summary(dirs["logs"] / "run_summary.txt", cfg, result["split"], dirs["run"])
    if visualize:
        result["visualize"] = visualize_dataset.run(cfg, dirs)
    result["train_nuclei"] = train_yolo_nuclei.run(cfg, dirs)
    result["train_boundary"] = train_cell_boundary.run(cfg, dirs)
    result["infer"] = infer_pipeline.run(cfg, dirs)
    result["matching"] = match_nucleus_boundary.run(cfg, dirs)
    result["morphology"] = compute_morphology.run(cfg, dirs)
    result["evaluate"] = evaluate_pipeline.run(cfg, dirs)
    write_json(dirs["metrics"] / "all_pipeline_summary.json", result)
    update_report(cfg, dirs, result)
    return result


def update_report(cfg: dict[str, Any], dirs: dict[str, Path], result: dict[str, Any]) -> None:
    preview_dir = dirs["run"] / "dataset_preview"
    plot_dir = dirs["run"] / "plots"
    images = [
        preview_dir / "train_tiles_grid.png",
        preview_dir / "val_tiles_grid.png",
        preview_dir / "class_examples_grid.png",
        preview_dir / "class_distribution.png",
        preview_dir / "batch_distribution.png",
    ]
    images.extend(sorted((dirs["overlays"]).glob("**/*overlay.png"))[:8])
    images.extend(sorted(plot_dir.glob("*.png"))[:8])
    write_html_report(
        dirs["run"] / "training_report.html",
        "Adrenal H&E Morphology Training Report",
        {
            "Experiment": {
                "name": cfg.get("experiment", {}).get("name", ""),
                "run_dir": dirs["run"],
                "config": cfg.get("_config_path", ""),
            },
            "Latest Result": result,
            "Figures": [path for path in images if path.exists()],
            "Recommendation": result.get("evaluate", {}).get("recommendation", result.get("morphology", {}).get("recommendation", "inspect overlays and QC CSVs")),
        },
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    dirs = initialize_run(cfg)
    current_stage = args.stage
    try:
        result = run_stage(args.stage, cfg, dirs, visualize=args.visualize)
        split_summary = result.get("split", result) if isinstance(result, dict) else {}
        write_run_summary(dirs["logs"] / "run_summary.txt", cfg, split_summary if isinstance(split_summary, dict) else {}, dirs["run"])
        write_json(dirs["metrics"] / "stage_result.json", result)
        update_report(cfg, dirs, result if isinstance(result, dict) else {"result": result})
        print(f"Pipeline stage '{args.stage}' complete.")
        print(f"Run folder: {dirs['run']}")
    except Exception as exc:
        write_error_report(dirs["logs"] / "error_report.txt", exc, current_stage)
        print(f"Pipeline stage '{args.stage}' failed. See: {dirs['logs'] / 'error_report.txt'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
