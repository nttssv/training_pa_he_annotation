#!/usr/bin/env python3
"""Create one self-contained combined CellSeg1 training batch.

The output folder contains real copied files, not symlinks, so it can be used as
a single unambiguous dataset root after cloning the repository on a GPU cluster.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cellseg1_cgh_p2_combined_41_full"

BATCHES = [
    {
        "name": "batch1",
        "root": ROOT,
        "description": "Original curated CellSeg1 CGH P2 training batch.",
    },
    {
        "name": "batch2",
        "root": ROOT / "cellseg1_cgh_p2_second_batch",
        "description": "PNG-derived compact-cell supplemental batch.",
    },
]

MANIFEST_FIELDS = [
    "tile_id",
    "tile_name",
    "image_file",
    "mask_file",
    "width",
    "height",
    "trainable_instances",
    "clear_trainable",
    "compact_trainable",
    "edge_or_invalid_boundary_ignore",
    "uncertain_ignore_regions",
    "nuclei_total",
    "nuclei_in_tile",
    "stroma_regions_intersecting_tile",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_asset(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(source, dest)


def copy_csv_or_symlink(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(source, dest)


def clean_output() -> None:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        shutil.rmtree(OUTPUT)
    for rel in [
        "train/images",
        "train/masks",
        "auxiliary_masks",
        "semantic_masks",
        "previews",
        "per_tile_instances",
    ]:
        (OUTPUT / rel).mkdir(parents=True, exist_ok=True)


def remove_appledouble_files(path: Path) -> int:
    removed = 0
    for sidecar in path.rglob("._*"):
        if sidecar.is_file() or sidecar.is_symlink():
            sidecar.unlink()
            removed += 1
    return removed


def main() -> int:
    clean_output()

    combined_manifest: list[dict[str, object]] = []
    combined_cells: list[dict[str, str]] = []
    combined_qc: list[dict[str, str]] = []
    membership: list[dict[str, object]] = []
    batch_stats: list[dict[str, object]] = []

    seen_tile_ids: set[str] = set()

    for batch in BATCHES:
        batch_name = str(batch["name"])
        batch_root = Path(batch["root"])
        manifest_rows = read_csv(batch_root / "dataset_manifest.csv")
        cell_rows = read_csv(batch_root / "cell_instances.csv")
        qc_rows = read_csv(batch_root / "boundary_qc.csv")

        stats = {
            "batch": batch_name,
            "description": batch["description"],
            "source_root": str(batch_root.relative_to(ROOT))
            if batch_root != ROOT
            else ".",
            "tile_count": len(manifest_rows),
            "trainable_instances": sum(
                int(row["trainable_instances"]) for row in manifest_rows
            ),
            "clear_trainable": sum(
                int(row["clear_trainable"]) for row in manifest_rows
            ),
            "compact_trainable": sum(
                int(row["compact_trainable"]) for row in manifest_rows
            ),
            "stroma_regions_intersecting_tile": sum(
                int(row["stroma_regions_intersecting_tile"]) for row in manifest_rows
            ),
            "cell_instances_rows": len(cell_rows),
            "boundary_qc_rows": len(qc_rows),
        }
        batch_stats.append(stats)

        for row in manifest_rows:
            tile_id = row["tile_id"]
            if tile_id in seen_tile_ids:
                raise ValueError(f"Duplicate tile_id in combined batch: {tile_id}")
            seen_tile_ids.add(tile_id)

            image_source = batch_root / row["image_file"]
            mask_source = batch_root / row["mask_file"]
            image_dest = OUTPUT / "train/images" / f"{tile_id}.png"
            mask_dest = OUTPUT / "train/masks" / f"{tile_id}.png"
            copy_asset(image_source, image_dest)
            copy_asset(mask_source, mask_dest)

            manifest_row = {field: row[field] for field in MANIFEST_FIELDS}
            manifest_row["image_file"] = f"train/images/{tile_id}.png"
            manifest_row["mask_file"] = f"train/masks/{tile_id}.png"
            combined_manifest.append(manifest_row)

            membership.append(
                {
                    "tile_id": tile_id,
                    "tile_name": row["tile_name"],
                    "source_batch": batch_name,
                    "source_image_file": str(image_source.relative_to(ROOT)),
                    "source_mask_file": str(mask_source.relative_to(ROOT)),
                }
            )

            instance_csv = batch_root / f"{tile_id}_instances.csv"
            if instance_csv.exists():
                copy_csv_or_symlink(
                    instance_csv, OUTPUT / "per_tile_instances" / instance_csv.name
                )

        for directory in ["auxiliary_masks", "semantic_masks", "previews"]:
            source_dir = batch_root / directory
            if not source_dir.exists():
                continue
            for source in sorted(source_dir.glob("*")):
                if source.name.startswith("._") or not source.is_file():
                    continue
                copy_asset(source, OUTPUT / directory / source.name)

        combined_cells.extend(cell_rows)
        combined_qc.extend(qc_rows)

    combined_manifest.sort(key=lambda row: str(row["tile_id"]))
    combined_cells.sort(
        key=lambda row: (
            str(row["tile_id"]),
            int(row.get("instance_label") or 0),
            str(row.get("boundary_name") or ""),
        )
    )
    combined_qc.sort(
        key=lambda row: (
            str(row["tile_id"]),
            str(row.get("boundary_name") or ""),
            str(row.get("boundary_class") or ""),
        )
    )
    membership.sort(key=lambda row: str(row["tile_id"]))

    write_csv(OUTPUT / "dataset_manifest.csv", MANIFEST_FIELDS, combined_manifest)
    write_csv(
        OUTPUT / "cell_instances.csv",
        list(combined_cells[0].keys()) if combined_cells else [],
        combined_cells,
    )
    write_csv(
        OUTPUT / "boundary_qc.csv",
        list(combined_qc[0].keys()) if combined_qc else [],
        combined_qc,
    )
    write_csv(
        OUTPUT / "batch_membership.csv",
        [
            "tile_id",
            "tile_name",
            "source_batch",
            "source_image_file",
            "source_mask_file",
        ],
        membership,
    )

    totals = {
        "tile_count": len(combined_manifest),
        "trainable_instances": sum(
            int(row["trainable_instances"]) for row in combined_manifest
        ),
        "clear_trainable": sum(int(row["clear_trainable"]) for row in combined_manifest),
        "compact_trainable": sum(
            int(row["compact_trainable"]) for row in combined_manifest
        ),
        "stroma_regions_intersecting_tile": sum(
            int(row["stroma_regions_intersecting_tile"]) for row in combined_manifest
        ),
        "cell_instances_rows": len(combined_cells),
        "boundary_qc_rows": len(combined_qc),
    }
    totals["clear_percent"] = round(
        100.0 * totals["clear_trainable"] / totals["trainable_instances"], 3
    )
    totals["compact_percent"] = round(
        100.0 * totals["compact_trainable"] / totals["trainable_instances"], 3
    )

    summary = {
        "source_type": "combined_cellseg1_cgh_p2_batch",
        "storage": "self_contained_copied_assets",
        "batches": batch_stats,
        **totals,
    }
    (OUTPUT / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Combined CGH P2 CellSeg1 Training Batch

This directory combines the original CGH P2 training batch and the second
PNG-derived compact-cell supplemental batch into one self-contained training
entrypoint.

The image, mask, auxiliary, semantic, and preview PNG files are copied into this
folder as real files, not symlinks. Use this folder as the dataset root on the
GPU cluster.

## Summary

- Tiles: {totals['tile_count']}
- Trainable cell instances: {totals['trainable_instances']}
- Clear cells: {totals['clear_trainable']} ({totals['clear_percent']}%)
- Compact cells: {totals['compact_trainable']} ({totals['compact_percent']}%)
- Stroma regions: {totals['stroma_regions_intersecting_tile']}
- Cell instance CSV rows: {totals['cell_instances_rows']}
- Boundary QC CSV rows: {totals['boundary_qc_rows']}

## Main Files

- `dataset_manifest.csv`
- `cell_instances.csv`
- `boundary_qc.csv`
- `batch_membership.csv`
- `train/images/`
- `train/masks/`
- `auxiliary_masks/`
- `semantic_masks/`
- `previews/`

Use this directory as the dataset root on the GPU cluster after cloning the repo.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    removed_appledouble = remove_appledouble_files(OUTPUT)
    summary["appledouble_cleanup"] = {
        "performed": True,
        "removed_before_summary_write": removed_appledouble,
    }
    (OUTPUT / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    summary["appledouble_cleanup"]["removed_after_summary_write"] = (
        remove_appledouble_files(OUTPUT)
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
