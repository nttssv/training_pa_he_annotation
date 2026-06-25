"""Compute measurement-grade adrenal cell morphology readouts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from training.match_nucleus_boundary import match_cells
from training.utils.io import read_csv, write_csv, write_json
from training.utils.metrics import safe_ratio


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def summarize(rows: list[dict[str, Any]], group_key: str, pixel_size_um: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "") or "unknown")].append(row)

    out: list[dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        valid = [row for row in items if row.get("qc_status") == "valid"]
        uncertain = [row for row in items if row.get("qc_status") != "valid"]
        edge_reject = [row for row in items if "edge" in str(row.get("qc_reason", ""))]
        clear = [row for row in valid if row.get("phenotype") == "clear"]
        compact = [row for row in valid if row.get("phenotype") == "compact"]
        area_scale = pixel_size_um**2
        out.append(
            {
                group_key: group,
                "total_cells": len(items),
                "valid_cells": len(valid),
                "uncertain_cells": len(uncertain),
                "valid_cell_rate": safe_ratio(len(valid), len(items)),
                "uncertain_cell_rate": safe_ratio(len(uncertain), len(items)),
                "edge_rejection_rate": safe_ratio(len(edge_reject), len(items)),
                "clear_cell_fraction": safe_ratio(len(clear), len(valid)),
                "compact_cell_fraction": safe_ratio(len(compact), len(valid)),
                "mean_nuclear_area_um2": safe_ratio(sum(_float(row, "nucleus_area_px") for row in valid) * area_scale, len(valid)),
                "mean_cell_area_um2": safe_ratio(sum(_float(row, "cell_area_px") for row in valid) * area_scale, len(valid)),
                "mean_cytoplasm_area_um2": safe_ratio(sum(_float(row, "cytoplasm_area_px") for row in valid) * area_scale, len(valid)),
                "mean_n_c_ratio": safe_ratio(sum(_float(row, "n_c_ratio") for row in valid), len(valid)),
            }
        )
    return out


def compute_morphology(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    matched_path = dirs["csv"] / "matched_cells.csv"
    if matched_path.exists():
        rows = read_csv(matched_path)
    else:
        rows = match_cells(cfg, dirs)
    pixel_size = float(cfg.get("measurement", {}).get("pixel_size_um", 1.0))
    area_scale = pixel_size**2

    cell_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["nucleus_area_um2"] = _float(row, "nucleus_area_px") * area_scale
        enriched["cell_area_um2"] = _float(row, "cell_area_px") * area_scale
        enriched["cytoplasm_area_um2"] = _float(row, "cytoplasm_area_px") * area_scale
        cell_rows.append(enriched)

    tile_summary = summarize(cell_rows, "tile_id", pixel_size)
    case_summary = summarize(cell_rows, "case_id", pixel_size) if any(row.get("case_id") for row in cell_rows) else []

    write_csv(dirs["csv"] / "morphology_cell_level.csv", cell_rows)
    write_csv(dirs["csv"] / "morphology_tile_summary.csv", tile_summary)
    if case_summary:
        write_csv(dirs["csv"] / "morphology_case_summary.csv", case_summary)
    summary = {
        "cell_level_csv": str(dirs["csv"] / "morphology_cell_level.csv"),
        "tile_summary_csv": str(dirs["csv"] / "morphology_tile_summary.csv"),
        "case_summary_csv": str(dirs["csv"] / "morphology_case_summary.csv") if case_summary else "",
        "valid_cell_rate": safe_ratio(sum(1 for row in cell_rows if row.get("qc_status") == "valid"), len(cell_rows)),
        "uncertain_cell_rate": safe_ratio(sum(1 for row in cell_rows if row.get("qc_status") != "valid"), len(cell_rows)),
    }
    write_json(dirs["metrics"] / "morphology_summary.json", summary)
    return summary


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return compute_morphology(cfg, dirs)

