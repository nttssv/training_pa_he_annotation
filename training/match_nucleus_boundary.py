"""Match nuclei to cell boundaries and assign morphology QC status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.utils.geometry import (
    centroids_from_instance_mask,
    instance_area,
    instance_labels,
    overlap_fraction,
    read_mask,
    touches_edge,
)
from training.utils.io import discover_pairs, load_cell_class_map, dataset_paths, write_csv, write_json


def phenotype_from_class(boundary_class: str) -> str:
    lower = boundary_class.lower()
    if "compact" in lower:
        return "compact"
    if "clear" in lower:
        return "clear"
    return "unknown"


def match_cells(cfg: dict[str, Any], dirs: dict[str, Path]) -> list[dict[str, Any]]:
    paths = dataset_paths(cfg)
    class_map = load_cell_class_map(paths["metadata"])
    thresholds = cfg.get("qc", {})
    min_cell_area = int(thresholds.get("min_cell_area_px", 100))
    edge_margin = int(thresholds.get("edge_margin_px", 2))
    max_stroma_overlap = float(thresholds.get("max_stroma_overlap_fraction", 0.05))
    max_uncertain_overlap = float(thresholds.get("max_uncertain_overlap_fraction", 0.01))

    rows: list[dict[str, Any]] = []
    for tile in discover_pairs(cfg):
        if tile["mask_path"] is None:
            continue
        boundary_mask = read_mask(tile["mask_path"])
        if boundary_mask.size == 0:
            continue
        nucleus_mask = read_mask(tile["nucleus_mask_path"], boundary_mask.shape)
        stroma_mask = read_mask(tile["stroma_mask_path"], boundary_mask.shape) > 0
        uncertain_mask = read_mask(tile["uncertain_mask_path"], boundary_mask.shape) > 0
        edge_mask = read_mask(tile["edge_mask_path"], boundary_mask.shape) > 0
        nuclei = centroids_from_instance_mask(nucleus_mask)

        for cell_id in instance_labels(boundary_mask):
            binary = boundary_mask == cell_id
            cell_area = instance_area(boundary_mask, cell_id)
            inside = [
                nucleus
                for nucleus in nuclei
                if binary[int(round(nucleus["centroid_y"])), int(round(nucleus["centroid_x"]))]
            ]
            nucleus_area = int(sum(item["area_px"] for item in inside))
            n_nuclei = len(inside)
            stroma_overlap = overlap_fraction(binary, stroma_mask)
            uncertain_overlap = overlap_fraction(binary, uncertain_mask | edge_mask)
            edge_touch = touches_edge(binary, edge_margin)

            reasons: list[str] = []
            if n_nuclei != 1:
                reasons.append("zero_nuclei" if n_nuclei == 0 else "multiple_nuclei")
            if cell_area < min_cell_area:
                reasons.append("too_small")
            if edge_touch:
                reasons.append("edge_touching")
            if stroma_overlap > max_stroma_overlap:
                reasons.append("stroma_overlap")
            if uncertain_overlap > max_uncertain_overlap:
                reasons.append("uncertain_or_edge_overlap")

            qc_status = "valid" if not reasons else "uncertain"
            cytoplasm_area = max(cell_area - nucleus_area, 0)
            row = {
                "tile_id": tile["tile_id"],
                "case_id": tile.get("case_id", ""),
                "batch_source": tile.get("batch_source", ""),
                "cell_id": cell_id,
                "phenotype": phenotype_from_class(class_map.get((tile["tile_id"], cell_id), "")),
                "n_nuclei": n_nuclei,
                "nucleus_ids": ";".join(str(item["label"]) for item in inside),
                "nucleus_area_px": nucleus_area,
                "cell_area_px": cell_area,
                "cytoplasm_area_px": cytoplasm_area,
                "n_c_ratio": float(nucleus_area / cytoplasm_area) if cytoplasm_area else float("nan"),
                "stroma_overlap_fraction": stroma_overlap,
                "uncertain_overlap_fraction": uncertain_overlap,
                "edge_touching": str(edge_touch).lower(),
                "qc_status": qc_status,
                "qc_reason": ";".join(reasons),
            }
            rows.append(row)
    write_csv(dirs["csv"] / "matched_cells.csv", rows)
    summary = {
        "matched_cells": len(rows),
        "valid_cells": sum(1 for row in rows if row["qc_status"] == "valid"),
        "uncertain_cells": sum(1 for row in rows if row["qc_status"] != "valid"),
    }
    write_json(dirs["metrics"] / "matching_summary.json", summary)
    return rows


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    rows = match_cells(cfg, dirs)
    return {
        "matched_cells_csv": str(dirs["csv"] / "matched_cells.csv"),
        "matched_cells": len(rows),
    }

