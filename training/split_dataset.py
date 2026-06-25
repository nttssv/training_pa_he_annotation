"""Create tile-level train/validation manifests."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from training.utils.io import PipelineError, discover_pairs, image_size, write_csv, write_json


def split_pairs(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [row for row in discover_pairs(cfg) if row["mask_path"] is not None]
    if not pairs:
        raise PipelineError("dataset", "No image/mask pairs available for train/validation split")
    pairs = sorted(pairs, key=lambda row: row["tile_id"])
    split_cfg = cfg.get("split", {})
    val_ids = set(split_cfg.get("val_tile_ids") or [])
    if val_ids:
        val = [row for row in pairs if row["tile_id"] in val_ids]
        train = [row for row in pairs if row["tile_id"] not in val_ids]
    elif split_cfg.get("val_every"):
        every = int(split_cfg["val_every"])
        val = [row for row in pairs if tile_number(row["tile_id"]) % every == 0]
        train = [row for row in pairs if row not in val]
    else:
        fraction = float(split_cfg.get("val_fraction", 0.2))
        seed = int(cfg.get("experiment", {}).get("seed", 42))
        shuffled = pairs[:]
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * fraction)))
        val_ids = {row["tile_id"] for row in shuffled[:n_val]}
        val = [row for row in pairs if row["tile_id"] in val_ids]
        train = [row for row in pairs if row["tile_id"] not in val_ids]
    if not train or not val:
        raise PipelineError("split", f"Invalid split: train={len(train)} val={len(val)}")
    return train, val


def tile_number(tile_id: str) -> int:
    try:
        return int(tile_id.rsplit("_", 1)[1])
    except Exception:
        return 0


def manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        width, height = image_size(row["image_path"])
        output.append(
            {
                "tile_id": row["tile_id"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "nucleus_mask_path": row["nucleus_mask_path"] or "",
                "stroma_mask_path": row["stroma_mask_path"] or "",
                "uncertain_mask_path": row["uncertain_mask_path"] or "",
                "edge_mask_path": row["edge_mask_path"] or "",
                "batch_source": row["batch_source"],
                "case_id": row["case_id"],
                "width": width,
                "height": height,
            }
        )
    return output


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    train, val = split_pairs(cfg)
    train_rows = manifest_rows(train)
    val_rows = manifest_rows(val)
    write_csv(dirs["csv"] / "train_manifest.csv", train_rows)
    write_csv(dirs["csv"] / "val_manifest.csv", val_rows)
    summary = {
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "train_tiles": [row["tile_id"] for row in train_rows],
        "val_tiles": [row["tile_id"] for row in val_rows],
    }
    write_json(dirs["metrics"] / "split_summary.json", summary)
    return summary

