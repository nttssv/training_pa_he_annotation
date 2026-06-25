"""Generate dataset previews before training."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from training.split_dataset import split_pairs
from training.utils.io import class_distribution_from_metadata, dataset_paths, write_json
from training.utils.visualization import bar_chart, make_image_grid, overlay_masks, write_html_report


def generate_dataset_visualizations(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    train, val = split_pairs(cfg)
    paths = dataset_paths(cfg)
    preview_dir = dirs["run"] / "dataset_preview"
    overlay_dir = preview_dir / "annotation_overlay_examples"
    preview_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    max_tiles = int(cfg.get("visualization", {}).get("max_preview_tiles", 16))
    train_grid = make_image_grid(
        [(row["image_path"], row["tile_id"]) for row in train[:max_tiles]],
        preview_dir / "train_tiles_grid.png",
    )
    val_grid = make_image_grid(
        [(row["image_path"], row["tile_id"]) for row in val[:max_tiles]],
        preview_dir / "val_tiles_grid.png",
    )

    overlay_paths: list[Path] = []
    for row in (train + val)[: int(cfg.get("visualization", {}).get("max_overlay_examples", 12))]:
        tile_id = row["tile_id"]
        out_path = overlay_dir / f"{tile_id}_overlay.png"
        overlay_masks(
            image_path=row["image_path"],
            out_path=out_path,
            boundary_mask=row["mask_path"],
            nucleus_mask=row["nucleus_mask_path"],
            clear_mask=paths["auxiliary_masks"] / f"{tile_id}_gt_clear_boundary_all.png",
            compact_mask=paths["auxiliary_masks"] / f"{tile_id}_gt_compact_boundary_all.png",
            uncertain_mask=row["uncertain_mask_path"],
            edge_mask=row["edge_mask_path"],
            title=tile_id,
        )
        overlay_paths.append(out_path)

    class_examples_grid = make_image_grid(
        [(path, path.stem.replace("_overlay", "")) for path in overlay_paths[:max_tiles]],
        preview_dir / "class_examples_grid.png",
    )

    class_counts = class_distribution_from_metadata(paths["metadata"])
    if not class_counts:
        class_counts = {"boundary_instances": sum(1 for row in train + val if row["mask_path"] is not None)}
    class_plot = bar_chart(class_counts, preview_dir / "class_distribution.png", "Class distribution")

    batch_counts = Counter(row.get("batch_source") or "unknown" for row in train + val)
    batch_plot = bar_chart(dict(batch_counts), preview_dir / "batch_distribution.png", "Batch distribution")

    report = write_html_report(
        dirs["run"] / "training_report.html",
        "Adrenal H&E Morphology Training Report",
        {
            "Dataset Preview": {
                "train_images": len(train),
                "val_images": len(val),
                "dataset_root": cfg.get("paths", {}).get("dataset_root", ""),
            },
            "Preview Figures": [train_grid, val_grid, class_examples_grid, class_plot, batch_plot, *overlay_paths[:6]],
            "QC Rule Reminder": "Valid cell requires exactly one nucleus inside a cell boundary; zero, multiple, edge-cut, stroma/vessel-overlap, or ambiguous phenotype cells are uncertain.",
        },
    )
    summary = {
        "preview_dir": str(preview_dir),
        "train_grid": str(train_grid),
        "val_grid": str(val_grid),
        "class_examples_grid": str(class_examples_grid),
        "class_distribution": str(class_plot),
        "batch_distribution": str(batch_plot),
        "overlay_examples": [str(path) for path in overlay_paths],
        "training_report": str(report),
    }
    write_json(dirs["metrics"] / "dataset_visualization_summary.json", summary)
    return summary


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return generate_dataset_visualizations(cfg, dirs)
