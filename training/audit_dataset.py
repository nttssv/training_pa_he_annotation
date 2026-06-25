"""Audit adrenal H&E training data before model training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.utils.io import (
    PipelineError,
    class_distribution_from_metadata,
    dataset_paths,
    discover_pairs,
    image_size,
    write_csv,
    write_json,
)


def audit_dataset(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    paths = dataset_paths(cfg)
    pairs = discover_pairs(cfg)
    if not pairs:
        raise PipelineError("dataset", f"No images found in configured dataset: {paths['images']}")

    image_names = [Path(row["image_path"]).name for row in pairs]
    mask_names = [Path(row["mask_path"]).name for row in pairs if row["mask_path"] is not None]
    duplicate_images = [name for name, count in Counter(image_names).items() if count > 1]
    duplicate_masks = [name for name, count in Counter(mask_names).items() if count > 1]

    missing_masks: list[str] = []
    mismatches: list[str] = []
    batch_counts: Counter[str] = Counter()
    for row in pairs:
        batch_counts[row.get("batch_source") or "unknown"] += 1
        mask_path = row["mask_path"]
        if mask_path is None:
            missing_masks.append(row["tile_id"])
            continue
        img_size = image_size(row["image_path"])
        mask_size = image_size(mask_path)
        if img_size != mask_size:
            mismatches.append(f"{row['tile_id']} image={img_size} mask={mask_size}")

    class_distribution = class_distribution_from_metadata(paths["metadata"])
    summary = {
        "dataset_root": str(paths["root"]),
        "number_of_images": len(pairs),
        "number_of_masks_or_labels": len(mask_names),
        "missing_image_mask_pairs": len(missing_masks),
        "missing_mask_tile_ids": ";".join(missing_masks),
        "duplicate_filenames": len(duplicate_images) + len(duplicate_masks),
        "duplicate_image_filenames": ";".join(duplicate_images),
        "duplicate_mask_filenames": ";".join(duplicate_masks),
        "image_mask_size_mismatches": len(mismatches),
        "image_mask_size_mismatch_details": ";".join(mismatches[:50]),
        "batch_source_distribution": json.dumps(dict(batch_counts), sort_keys=True),
        "class_distribution": json.dumps(class_distribution, sort_keys=True),
    }
    write_csv(dirs["csv"] / "dataset_summary.csv", [summary])
    write_csv(
        dirs["csv"] / "dataset_tiles.csv",
        [
            {
                "tile_id": row["tile_id"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"] or "",
                "nucleus_mask_path": row["nucleus_mask_path"] or "",
                "batch_source": row["batch_source"],
                "case_id": row["case_id"],
            }
            for row in pairs
        ],
    )
    write_json(dirs["metrics"] / "dataset_summary.json", summary)
    return summary


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return audit_dataset(cfg, dirs)

