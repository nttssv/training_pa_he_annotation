"""Evaluate boundary masks and morphology QC outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.compute_morphology import compute_morphology
from training.utils.io import read_json, write_json


def evaluate(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    morphology = compute_morphology(cfg, dirs)
    matching_path = dirs["metrics"] / "matching_summary.json"
    matching = read_json(matching_path) if matching_path.exists() else {}
    summary = {
        "valid_cell_rate": morphology.get("valid_cell_rate", ""),
        "uncertain_cell_rate": morphology.get("uncertain_cell_rate", ""),
        "matched_cells": matching.get("matched_cells", ""),
        "valid_cells": matching.get("valid_cells", ""),
        "recommendation": recommendation(morphology),
    }
    write_json(dirs["metrics"] / "evaluation_summary.json", summary)
    return summary


def recommendation(morphology: dict[str, Any]) -> str:
    try:
        valid = float(morphology.get("valid_cell_rate", 0))
        uncertain = float(morphology.get("uncertain_cell_rate", 1))
    except Exception:
        return "check annotations"
    if valid >= 0.75 and uncertain <= 0.25:
        return "continue training or run full inference"
    if valid < 0.5:
        return "check annotations and nucleus-boundary matching"
    return "continue training and inspect QC overlays"


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return evaluate(cfg, dirs)

