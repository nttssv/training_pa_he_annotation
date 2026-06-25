#!/usr/bin/env python3
"""Create 512x512 semantic tiles from a raw image and colored annotation overlay.

Mask convention:
- 0: stroma, from yellow annotation pixels
- 1: compact cell, from green annotation pixels
- 255: ignore
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IGNORE_VALUE = 255
STROMA_VALUE = 0
COMPACT_VALUE = 1


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or value.lower() == "auto":
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--annotation-crop must be auto or x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--annotation-crop must have x1>x0 and y1>y0")
    return x0, y0, x1, y1


def largest_contiguous_run(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    breaks = np.where(np.diff(values) > 1)[0] + 1
    runs = np.split(values, breaks)
    return max(runs, key=len)


def auto_content_crop(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main annotated image area inside a screenshot with margins/text."""
    non_background = ~(
        (rgb[:, :, 0] > 235) & (rgb[:, :, 1] > 235) & (rgb[:, :, 2] > 235)
    )
    row_frac = non_background.mean(axis=1)
    col_frac = non_background.mean(axis=0)
    rows = largest_contiguous_run(np.where(row_frac > 0.50)[0])
    cols = largest_contiguous_run(np.where(col_frac > 0.50)[0])
    if len(rows) == 0 or len(cols) == 0:
        ys, xs = np.where(non_background)
        rows = ys
        cols = xs

    return (
        max(int(cols.min()), 0),
        max(int(rows.min()), 0),
        min(int(cols.max()) + 1, rgb.shape[1]),
        min(int(rows.max()) + 1, rgb.shape[0]),
    )


def annotation_to_semantic_mask(annotation: Image.Image) -> np.ndarray:
    """Convert yellow/green annotation colors to a 0/1/255 semantic mask."""
    arr = np.array(annotation.convert("RGB"), dtype=np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    yellow = (
        (r > 170)
        & (g > 145)
        & (b < 150)
        & (r > b + 45)
        & (g > b + 35)
    )
    green = (
        (g > 85)
        & (r < 115)
        & (b < 170)
        & (g > r + 25)
        & (g > b - 10)
    )

    mask = np.full(arr.shape[:2], IGNORE_VALUE, dtype=np.uint8)
    mask[yellow] = STROMA_VALUE
    mask[green] = COMPACT_VALUE
    return mask


def pad_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    image_arr = np.array(image.convert("RGB"))
    height, width = mask.shape
    pad_bottom = max(tile_size - height, 0)
    pad_right = max(tile_size - width, 0)

    if pad_bottom or pad_right:
        image_pad_mode = "reflect" if height > 1 and width > 1 else "edge"
        image_arr = np.pad(
            image_arr,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode=image_pad_mode,
        )
        mask = np.pad(
            mask,
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=IGNORE_VALUE,
        )

    return image_arr, mask, {"pad_bottom": pad_bottom, "pad_right": pad_right}


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_stats(mask_tile: np.ndarray) -> dict[str, float | int]:
    total_pixels = int(mask_tile.size)
    valid = mask_tile != IGNORE_VALUE
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        stroma_percent = 0.0
        compact_percent = 0.0
    else:
        stroma_percent = 100.0 * float((mask_tile[valid] == STROMA_VALUE).sum()) / valid_pixels
        compact_percent = 100.0 * float((mask_tile[valid] == COMPACT_VALUE).sum()) / valid_pixels
    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_percent": 100.0 * valid_pixels / total_pixels,
        "stroma_percent": stroma_percent,
        "compact_percent": compact_percent,
    }


def save_preview(image_arr: np.ndarray, mask: np.ndarray, path: Path) -> None:
    preview = image_arr.astype(np.float32).copy()
    stroma = mask == STROMA_VALUE
    compact = mask == COMPACT_VALUE
    preview[stroma] = preview[stroma] * 0.45 + np.array([255, 230, 0]) * 0.55
    preview[compact] = preview[compact] * 0.45 + np.array([0, 180, 80]) * 0.55
    Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8), mode="RGB").save(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_tiles(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    source_id = args.source_id
    image_path = Path(args.image).resolve()
    annotation_path = Path(args.annotation).resolve()

    image = Image.open(image_path).convert("RGB")
    annotation_rgb = np.array(Image.open(annotation_path).convert("RGB"))
    crop = parse_crop(args.annotation_crop)
    if crop is None:
        crop = auto_content_crop(annotation_rgb)
    x0, y0, x1, y1 = crop
    annotation_crop = Image.fromarray(annotation_rgb[y0:y1, x0:x1, :], mode="RGB")
    annotation_resized = annotation_crop.resize(image.size, Image.Resampling.BICUBIC)
    full_mask = annotation_to_semantic_mask(annotation_resized)

    padded_image, padded_mask, padding = pad_image_and_mask(image, full_mask, args.tile_size)

    for rel in [
        "source_pairs",
        "train/images",
        "train/masks",
        "previews",
        "full_masks",
        "summaries",
    ]:
        (output_root / rel).mkdir(parents=True, exist_ok=True)

    source_dir = output_root / "source_pairs" / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, source_dir / "before.png")
    shutil.copy2(annotation_path, source_dir / "after_annotated.png")
    annotation_crop.save(source_dir / "after_annotated_auto_crop.png")
    annotation_resized.save(source_dir / "after_annotated_resized_to_image.png")
    Image.fromarray(full_mask, mode="L").save(output_root / "full_masks" / f"{source_id}_semantic.png")

    rows: list[dict[str, object]] = []
    skipped = 0
    tile_index = 0
    y_starts = tile_starts(padded_mask.shape[0], args.tile_size, args.stride)
    x_starts = tile_starts(padded_mask.shape[1], args.tile_size, args.stride)

    for y in y_starts:
        for x in x_starts:
            image_tile = padded_image[y : y + args.tile_size, x : x + args.tile_size, :]
            mask_tile = padded_mask[y : y + args.tile_size, x : x + args.tile_size]
            stats = tile_stats(mask_tile)
            if stats["valid_percent"] < args.min_valid_percent:
                skipped += 1
                continue

            tile_name = f"{source_id}_y{y:04d}_x{x:04d}"
            image_file = f"train/images/{tile_name}.png"
            mask_file = f"train/masks/{tile_name}.png"
            preview_file = f"previews/{tile_name}_overlay.png"

            Image.fromarray(image_tile, mode="RGB").save(output_root / image_file)
            Image.fromarray(mask_tile, mode="L").save(output_root / mask_file)
            save_preview(image_tile, mask_tile, output_root / preview_file)

            tile_index += 1
            rows.append(
                {
                    "tile_id": tile_name,
                    "source_id": source_id,
                    "image_file": image_file,
                    "mask_file": mask_file,
                    "preview_file": preview_file,
                    "x": x,
                    "y": y,
                    "width": args.tile_size,
                    "height": args.tile_size,
                    "valid_pixels": stats["valid_pixels"],
                    "total_pixels": stats["total_pixels"],
                    "valid_percent": f"{stats['valid_percent']:.4f}",
                    "stroma_percent": f"{stats['stroma_percent']:.4f}",
                    "compact_percent": f"{stats['compact_percent']:.4f}",
                }
            )

    write_csv(output_root / "tile_manifest.csv", rows)
    summary = {
        "source_id": source_id,
        "image": str(image_path),
        "annotation": str(annotation_path),
        "annotation_crop_xyxy": [x0, y0, x1, y1],
        "image_size": list(image.size),
        "padded_size": [int(padded_mask.shape[1]), int(padded_mask.shape[0])],
        "tile_size": args.tile_size,
        "stride": args.stride,
        "min_valid_percent": args.min_valid_percent,
        "padding": padding,
        "tiles_written": len(rows),
        "tiles_skipped_low_valid": skipped,
        "mask_values": {
            "stroma": STROMA_VALUE,
            "compact_cell": COMPACT_VALUE,
            "ignore": IGNORE_VALUE,
        },
        "full_mask_valid_percent": tile_stats(padded_mask)["valid_percent"],
        "full_mask_stroma_percent": tile_stats(padded_mask)["stroma_percent"],
        "full_mask_compact_percent": tile_stats(padded_mask)["compact_percent"],
        "tiles": rows,
    }
    with (output_root / "summaries" / f"{source_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Raw/before image")
    parser.add_argument("--annotation", required=True, help="After image with colored annotations")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--annotation-crop", default="auto", help="auto or x0,y0,x1,y1")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-valid-percent", type=float, default=5.0)
    args = parser.parse_args()

    summary = build_tiles(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
