#!/usr/bin/env python3
"""Validate the CGH P2 training-data export before publishing."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Pillow is required: python3 -m pip install pillow") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_csv_rows(path: Path) -> int:
    return len(read_csv(path))


def validate_yolo_labels(root: Path, errors: list[str]) -> tuple[int, int]:
    yolo_root = root / "yolo_seg_dataset"
    data_yaml = yolo_root / "data.yaml"
    if not data_yaml.exists():
        errors.append("missing yolo_seg_dataset/data.yaml")
        return 0, 0

    image_files = sorted(
        p for p in (yolo_root / "images").glob("*/*.png") if not p.name.startswith("._")
    )
    label_count = 0
    for image_file in image_files:
        split = image_file.parent.name
        label_file = yolo_root / "labels" / split / f"{image_file.stem}.txt"
        if not label_file.exists():
            errors.append(f"missing YOLO label for {image_file.relative_to(root)}")
            continue
        with label_file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7 or len(parts) % 2 == 0:
                    errors.append(
                        f"invalid YOLO polygon line {label_file.relative_to(root)}:{line_no}"
                    )
                    continue
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(
                        f"non-numeric YOLO values {label_file.relative_to(root)}:{line_no}"
                    )
                    continue
                if cls < 0:
                    errors.append(
                        f"negative YOLO class {label_file.relative_to(root)}:{line_no}"
                    )
                if any(v < 0.0 or v > 1.0 for v in coords):
                    errors.append(
                        f"YOLO coordinate outside 0..1 {label_file.relative_to(root)}:{line_no}"
                    )
                label_count += 1
    return len(image_files), label_count


def validate(root: Path, expected_tiles: int | None) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = root / "dataset_manifest.csv"
    cell_instances_path = root / "cell_instances.csv"
    boundary_qc_path = root / "boundary_qc.csv"

    for path in [manifest_path, cell_instances_path, boundary_qc_path]:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(root)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    manifest = read_csv(manifest_path)
    tile_count = len(manifest)
    if expected_tiles is not None and tile_count != expected_tiles:
        errors.append(f"manifest rows {tile_count} != expected tiles {expected_tiles}")

    total_instances = 0
    for row in manifest:
        tile_id = row["tile_id"]
        expected_instances = int(row["trainable_instances"])
        total_instances += expected_instances

        image_path = root / row["image_file"]
        mask_path = root / row["mask_file"]
        per_tile_csv = root / f"{tile_id}_instances.csv"

        if not image_path.exists():
            errors.append(f"{tile_id}: missing image {row['image_file']}")
            continue
        if not mask_path.exists():
            errors.append(f"{tile_id}: missing mask {row['mask_file']}")
            continue
        if not per_tile_csv.exists():
            errors.append(f"{tile_id}: missing {tile_id}_instances.csv")
            continue

        expected_size = (int(row["width"]), int(row["height"]))
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != expected_size:
                errors.append(f"{tile_id}: image size {image.size} != {expected_size}")
            if mask.size != expected_size:
                errors.append(f"{tile_id}: mask size {mask.size} != {expected_size}")
            labels = set(mask.getdata())
            positive_labels = labels - {0}
            expected_labels = set(range(1, expected_instances + 1))
            if positive_labels != expected_labels:
                missing = sorted(expected_labels - positive_labels)
                extra = sorted(positive_labels - expected_labels)
                errors.append(
                    f"{tile_id}: mask labels not contiguous; missing={missing}, extra={extra}"
                )

        per_tile_rows = count_csv_rows(per_tile_csv)
        if per_tile_rows != expected_instances:
            errors.append(
                f"{tile_id}: per-tile CSV rows {per_tile_rows} != {expected_instances}"
            )

    cell_rows = read_csv(cell_instances_path)
    cell_counts = Counter(row["tile_id"] for row in cell_rows)
    if len(cell_rows) != total_instances:
        errors.append(f"cell_instances rows {len(cell_rows)} != {total_instances}")
    for row in manifest:
        tile_id = row["tile_id"]
        expected_instances = int(row["trainable_instances"])
        if cell_counts[tile_id] != expected_instances:
            errors.append(
                f"{tile_id}: cell_instances rows {cell_counts[tile_id]} != {expected_instances}"
            )

    qc_rows = read_csv(boundary_qc_path)
    qc_included = [
        row for row in qc_rows if row["include_for_cellseg1"].strip().lower() == "true"
    ]
    qc_counts = Counter(row["tile_id"] for row in qc_included)
    if len(qc_included) != total_instances:
        errors.append(f"boundary_qc included rows {len(qc_included)} != {total_instances}")
    for row in manifest:
        tile_id = row["tile_id"]
        expected_instances = int(row["trainable_instances"])
        if qc_counts[tile_id] != expected_instances:
            errors.append(
                f"{tile_id}: boundary_qc included {qc_counts[tile_id]} != {expected_instances}"
            )

    appledouble_count = sum(1 for p in root.rglob("._*") if p.is_file())
    if appledouble_count:
        warnings.append(f"AppleDouble files present but ignored by git: {appledouble_count}")

    yolo_images, yolo_labels = validate_yolo_labels(root, errors)

    print("SUMMARY")
    print(f"tiles: {tile_count}")
    print(f"trainable_instances: {total_instances}")
    print(f"cell_instances_rows: {len(cell_rows)}")
    print(f"boundary_qc_included_rows: {len(qc_included)}")
    print(f"yolo_images: {yolo_images}")
    print(f"yolo_labels: {yolo_labels}")

    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if errors:
        print("ERRORS")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("VALIDATION PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="training data root")
    parser.add_argument("--expected-tiles", type=int, default=None)
    args = parser.parse_args()
    return validate(Path(args.root).resolve(), args.expected_tiles)


if __name__ == "__main__":
    sys.exit(main())
