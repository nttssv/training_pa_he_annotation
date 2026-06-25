"""Run two-model inference or export GT-backed predictions for QC smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.utils.geometry import instance_labels, read_mask
from training.utils.io import discover_pairs, write_json
from training.utils.visualization import overlay_masks


def infer(cfg: dict[str, Any], dirs: dict[str, Path], limit: int | None = None) -> dict[str, Any]:
    pairs = [row for row in discover_pairs(cfg) if row["mask_path"] is not None]
    if limit is not None:
        pairs = pairs[:limit]
    boundary_predictions: list[dict[str, Any]] = []
    nuclei_predictions: list[dict[str, Any]] = []
    for row in pairs:
        boundary_mask = read_mask(row["mask_path"])
        nucleus_mask = read_mask(row["nucleus_mask_path"], boundary_mask.shape if boundary_mask.size else None)
        boundary_predictions.append(
            {
                "tile_id": row["tile_id"],
                "source": "ground_truth_placeholder",
                "instance_count": len(instance_labels(boundary_mask)),
                "mask_path": str(row["mask_path"]),
            }
        )
        nuclei_predictions.append(
            {
                "tile_id": row["tile_id"],
                "source": "ground_truth_placeholder",
                "instance_count": len(instance_labels(nucleus_mask)),
                "mask_path": str(row["nucleus_mask_path"] or ""),
            }
        )
        if len(boundary_predictions) <= int(cfg.get("visualization", {}).get("max_overlay_examples", 12)):
            overlay_masks(
                row["image_path"],
                dirs["overlays"] / "inference" / f"{row['tile_id']}_prediction_overlay.png",
                boundary_mask=row["mask_path"],
                nucleus_mask=row["nucleus_mask_path"],
                title=f"{row['tile_id']} inference placeholder",
            )
    write_json(dirs["predictions"] / "boundary_predictions.json", boundary_predictions)
    write_json(dirs["predictions"] / "nuclei_predictions.json", nuclei_predictions)
    return {
        "boundary_predictions": str(dirs["predictions"] / "boundary_predictions.json"),
        "nuclei_predictions": str(dirs["predictions"] / "nuclei_predictions.json"),
        "tiles": len(pairs),
        "note": "Uses ground-truth masks as placeholder predictions unless model-specific inference is added/configured.",
    }


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return infer(cfg, dirs)

