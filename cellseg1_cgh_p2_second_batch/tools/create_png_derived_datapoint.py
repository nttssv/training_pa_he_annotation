#!/usr/bin/env python3
"""Create a PNG-derived training datapoint from before/after annotation screenshots.

This converter is intentionally conservative. It extracts colored human overlay
lines, fills only closed loops, excludes border-touching regions, and writes a
small CellSeg1/SAM-style datapoint for review before real training.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


SEMANTIC_CLEAR = 1
SEMANTIC_COMPACT = 2
SEMANTIC_UNCERTAIN = 3
SEMANTIC_STROMA = 4


@dataclass
class Instance:
    label: int
    class_key: str
    class_name: str
    boundary_name: str
    area_px: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or value.lower() == "auto":
        return None
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--annotated-crop must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--annotated-crop must have x1>x0 and y1>y0")
    return x0, y0, x1, y1


def parse_polygon(value: str) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for token in value.replace(";", " ").split():
        x_str, y_str = token.split(",", 1)
        points.append((int(round(float(x_str))), int(round(float(y_str)))))
    if len(points) < 3:
        raise ValueError("--extra-stroma-polygon requires at least 3 x,y points")
    return points


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1 in 512x512 output coords")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have x1>x0 and y1>y0")
    return x0, y0, x1, y1


def auto_content_crop(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the tissue/overlay rectangle in a screenshot with light margins."""
    non_white = ~(
        (rgb[:, :, 0] > 235) & (rgb[:, :, 1] > 235) & (rgb[:, :, 2] > 235)
    )
    row_frac = non_white.mean(axis=1)
    col_frac = non_white.mean(axis=0)
    rows = largest_contiguous_run(np.where(row_frac > 0.50)[0])
    cols = largest_contiguous_run(np.where(col_frac > 0.50)[0])
    if len(rows) == 0 or len(cols) == 0:
        ys, xs = np.where(non_white)
        rows = ys
        cols = xs
    pad = 1
    x0 = max(int(cols.min()) - pad, 0)
    y0 = max(int(rows.min()) - pad, 0)
    x1 = min(int(cols.max()) + 1 + pad, rgb.shape[1])
    y1 = min(int(rows.max()) + 1 + pad, rgb.shape[0])
    return x0, y0, x1, y1


def largest_contiguous_run(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    breaks = np.where(np.diff(values) > 1)[0] + 1
    runs = np.split(values, breaks)
    return max(runs, key=len)


def resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.array(image.resize((size, size), Image.Resampling.NEAREST)) > 0


def rasterize_polygons(polygons: list[list[tuple[int, int]]], size: int) -> np.ndarray:
    if not polygons:
        return np.zeros((size, size), dtype=bool)
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    return np.array(image) > 0


def extract_red_correction_mask(path: Path, size: int) -> np.ndarray:
    """Extract user-highlighted red correction areas from a preview screenshot."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    image = image.crop((x0, y0, x0 + side, y0 + side)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    arr = np.array(image).astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    mask = (r > 190) & (g < 150) & (b < 130) & (r > g + 45) & (r > b + 50)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)), iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    mask = ndi.binary_fill_holes(mask)

    labeled, count = ndi.label(mask)
    if count == 0:
        return np.zeros((size, size), dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count + 1):
        if sizes[label] >= 15:
            keep |= labeled == label
    return keep


def extract_orange_correction_mask(path: Path, size: int) -> np.ndarray:
    """Extract user-highlighted red/orange correction areas from a preview screenshot."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    image = image.crop((x0, y0, x0 + side, y0 + side)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    arr = np.array(image).astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    mask = (
        (r > 210)
        & (g < 130)
        & (b < 120)
        & (r > g + 75)
        & (r > b + 85)
    )
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)), iterations=1)
    mask = ndi.binary_dilation(mask, iterations=1)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=2)
    mask = ndi.binary_fill_holes(mask)

    labeled, count = ndi.label(mask)
    if count == 0:
        return np.zeros((size, size), dtype=bool)
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count + 1):
        if sizes[label] >= 15:
            keep |= labeled == label
    return keep


def trim_stroma_to_pink_tissue(
    stroma_mask: np.ndarray,
    image: Image.Image,
    *,
    max_value: float,
    min_saturation: float,
    min_pinkness: float,
    dark_value: float,
    dark_min_pinkness: float,
    close_iter: int,
    dilate: int,
    min_area: int,
    fill_holes_max_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain stroma to darker pink tissue and away from pale/white cell areas."""
    if not stroma_mask.any():
        empty = np.zeros_like(stroma_mask, dtype=bool)
        return empty, empty

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = (max_channel - min_channel) / (max_channel + 1e-6)
    pinkness = ((r + b) / 2.0) - g

    color_keep = (
        (max_channel < max_value)
        & (saturation > min_saturation)
        & (pinkness > min_pinkness)
    ) | ((max_channel < dark_value) & (pinkness > dark_min_pinkness))

    if close_iter:
        color_keep = ndi.binary_closing(
            color_keep, structure=np.ones((3, 3)), iterations=close_iter
        )
    if dilate:
        color_keep = ndi.binary_dilation(color_keep, iterations=dilate)

    trimmed = stroma_mask & color_keep
    labeled, count = ndi.label(trimmed)
    if count == 0:
        return np.zeros_like(stroma_mask, dtype=bool), color_keep

    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(trimmed, dtype=bool)
    for label in range(1, count + 1):
        if sizes[label] >= min_area:
            keep |= labeled == label

    if fill_holes_max_area > 0 and keep.any():
        holes = ndi.binary_fill_holes(keep) & ~keep
        hole_labels, hole_count = ndi.label(holes)
        if hole_count:
            hole_sizes = np.bincount(hole_labels.ravel())
            for label in range(1, hole_count + 1):
                if hole_sizes[label] <= fill_holes_max_area:
                    keep |= hole_labels == label
    return keep, color_keep


def disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0 or not mask.any():
        return mask.copy()

    output = mask.copy()
    holes = ndi.binary_fill_holes(output) & ~output
    labels, count = ndi.label(holes)
    if count == 0:
        return output

    sizes = np.bincount(labels.ravel())
    for label in range(1, count + 1):
        if sizes[label] <= max_area:
            output[labels == label] = True
    return output


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0 or not mask.any():
        return mask.copy()

    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def postprocess_stroma_mask(
    stroma_mask: np.ndarray,
    *,
    min_region_area: int,
    fill_holes_max_area: int,
    closing_radius: int,
    opening_radius: int,
    gaussian_sigma: float,
    threshold: float,
    area_change_tolerance: float,
    preserve_core_radius: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Smooth stroma masks while preserving area and connected tissue corridors."""
    original = stroma_mask.astype(bool)
    original_area = int(original.sum())
    original_components = int(ndi.label(original)[1])
    if original_area == 0:
        return original, {
            "enabled": True,
            "pixels_before": 0,
            "pixels_after": 0,
            "area_change_fraction": 0.0,
            "components_before": original_components,
            "components_after": 0,
            "selected_threshold": threshold,
        }

    cleaned = remove_small_components(original, min_region_area)
    cleaned = fill_small_holes(cleaned, fill_holes_max_area)

    close_structure = disk_structure(closing_radius)
    open_structure = disk_structure(opening_radius)

    if closing_radius > 0:
        cleaned = ndi.binary_closing(cleaned, structure=close_structure)
    if opening_radius > 0:
        cleaned = ndi.binary_opening(cleaned, structure=open_structure)
    cleaned = remove_small_components(cleaned, min_region_area)
    cleaned = fill_small_holes(cleaned, fill_holes_max_area)

    if preserve_core_radius > 0:
        core = ndi.binary_erosion(
            cleaned, structure=disk_structure(preserve_core_radius)
        )
    else:
        core = np.zeros_like(cleaned, dtype=bool)

    def smooth_at(candidate_threshold: float) -> np.ndarray:
        if gaussian_sigma > 0:
            score = ndi.gaussian_filter(cleaned.astype(np.float32), sigma=gaussian_sigma)
            smoothed = score >= candidate_threshold
        else:
            smoothed = cleaned.copy()
        smoothed |= core
        if closing_radius > 0:
            smoothed = ndi.binary_closing(smoothed, structure=close_structure)
        if opening_radius > 0:
            smoothed = ndi.binary_opening(smoothed, structure=open_structure)
        smoothed = fill_small_holes(smoothed, fill_holes_max_area)
        smoothed = remove_small_components(smoothed, min_region_area)
        return smoothed

    candidate_thresholds = [threshold]
    for delta in (0.05, 0.10, 0.15, 0.20):
        candidate_thresholds.extend([threshold - delta, threshold + delta])
    candidate_thresholds = [
        float(np.clip(value, 0.05, 0.95)) for value in candidate_thresholds
    ]

    best_mask = cleaned
    best_threshold = None
    best_area_delta = abs(int(cleaned.sum()) - original_area)
    for candidate_threshold in candidate_thresholds:
        candidate = smooth_at(candidate_threshold)
        candidate_area_delta = abs(int(candidate.sum()) - original_area)
        if candidate_area_delta < best_area_delta:
            best_mask = candidate
            best_threshold = candidate_threshold
            best_area_delta = candidate_area_delta

    output = best_mask
    area_change_fraction = (
        abs(int(output.sum()) - original_area) / float(original_area)
        if original_area
        else 0.0
    )
    if area_change_fraction > area_change_tolerance:
        output = cleaned
        best_threshold = None
        area_change_fraction = (
            abs(int(output.sum()) - original_area) / float(original_area)
            if original_area
            else 0.0
        )
    if area_change_fraction > area_change_tolerance:
        output = original
        best_threshold = None
        area_change_fraction = 0.0

    stats = {
        "enabled": True,
        "pixels_before": original_area,
        "pixels_after": int(output.sum()),
        "area_change_fraction": round(area_change_fraction, 6),
        "components_before": original_components,
        "components_after": int(ndi.label(output)[1]),
        "min_region_area": min_region_area,
        "fill_holes_max_area": fill_holes_max_area,
        "closing_radius": closing_radius,
        "opening_radius": opening_radius,
        "gaussian_sigma": gaussian_sigma,
        "requested_threshold": threshold,
        "selected_threshold": best_threshold,
        "area_change_tolerance": area_change_tolerance,
        "preserve_core_radius": preserve_core_radius,
    }
    return output, stats


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    breaks = np.where(np.diff(values) > 1)[0] + 1
    runs = np.split(values, breaks)
    return [(int(run[0]), int(run[-1])) for run in runs]


def fill_line_corridors(
    line_mask: np.ndarray,
    *,
    min_gap: int,
    max_gap: int,
    line_dilate: int,
    close_iter: int,
) -> np.ndarray:
    """Fill narrow open corridors bounded by two nearby colored annotation lines."""
    if max_gap <= 0 or not line_mask.any():
        return np.zeros_like(line_mask, dtype=bool)

    barrier = ndi.binary_dilation(line_mask, iterations=line_dilate)
    height, width = barrier.shape
    corridors = np.zeros_like(barrier, dtype=bool)

    for x in range(width):
        runs = contiguous_runs(np.where(barrier[:, x])[0])
        for (_, upper_end), (lower_start, _) in zip(runs, runs[1:]):
            gap = lower_start - upper_end - 1
            if min_gap <= gap <= max_gap:
                corridors[upper_end + 1 : lower_start, x] = True

    for y in range(height):
        runs = contiguous_runs(np.where(barrier[y, :])[0])
        for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
            gap = right_start - left_end - 1
            if min_gap <= gap <= max_gap:
                corridors[y, left_end + 1 : right_start] = True

    if close_iter:
        corridors = ndi.binary_closing(
            corridors, structure=np.ones((3, 3)), iterations=close_iter
        )
    corridors = ndi.binary_fill_holes(corridors)
    corridors = ndi.binary_opening(corridors, structure=np.ones((2, 2)), iterations=1)
    return corridors


def threshold_overlay_lines(rgb: np.ndarray) -> dict[str, np.ndarray]:
    arr = rgb.astype(np.int16)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green = (
        (g > 90)
        & (r < 95)
        & (b < 150)
        & (g > r + 35)
        & (g > b + 5)
    )
    yellow = (
        (r > 170)
        & (g > 155)
        & (b < 145)
        & (r > b + 55)
        & (g > b + 45)
    )
    blue = (
        (b > 145)
        & (r < 95)
        & (b > r + 70)
        & (b > g + 15)
    )

    return {
        "compact": green,
        "stroma": yellow,
        "clear": blue,
    }


def remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labeled]


def fill_closed_regions(
    line_mask: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    line_dilate: int,
    close_iter: int,
    assign_radius: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fill closed colored outlines and return binary union + per-instance masks."""
    if not line_mask.any():
        return np.zeros_like(line_mask, dtype=bool), []

    barrier = ndi.binary_dilation(line_mask, iterations=line_dilate)
    if close_iter:
        barrier = ndi.binary_closing(barrier, structure=np.ones((3, 3)), iterations=close_iter)
    barrier = remove_small(barrier, min_pixels=8)

    filled = ndi.binary_fill_holes(barrier)
    interiors = filled & ~barrier
    labeled, count = ndi.label(interiors)

    seeds = np.zeros_like(labeled, dtype=np.int32)
    next_label = 1
    for label in range(1, count + 1):
        component = labeled == label
        if not component.any():
            continue
        area = int(component.sum())
        ys, xs = np.where(component)
        touches_border = (
            ys.min() == 0
            or xs.min() == 0
            or ys.max() == component.shape[0] - 1
            or xs.max() == component.shape[1] - 1
        )
        if touches_border or area < min_area or area > max_area:
            continue
        seeds[component] = next_label
        next_label += 1

    if next_label == 1:
        return np.zeros_like(line_mask, dtype=bool), []

    labels = seeds.copy()
    expansion_zone = filled & (labels == 0)
    if expansion_zone.any():
        distances, nearest = ndi.distance_transform_edt(labels == 0, return_indices=True)
        nearest_labels = labels[tuple(nearest)]
        assign = expansion_zone & (distances <= assign_radius) & (nearest_labels > 0)
        labels[assign] = nearest_labels[assign]

    instances: list[np.ndarray] = []
    union = np.zeros_like(line_mask, dtype=bool)
    for label in range(1, next_label):
        component = labels == label
        area = int(component.sum())
        if area < min_area or area > max_area:
            continue
        instances.append(component)
        union |= component
    return union, instances


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    return mask & ~ndi.binary_erosion(mask)


def label_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary & (labels > 0)


def compose_instance_label_preview(image: Image.Image, instance_mask: np.ndarray) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    output = base.copy()
    labels = sorted(int(v) for v in np.unique(instance_mask) if int(v) != 0)
    for label in labels:
        rng = np.random.default_rng(label * 104729)
        color = rng.integers(40, 230, size=3).astype(np.float32)
        mask = instance_mask == label
        output[mask] = output[mask] * 0.50 + color * 0.50
    output[label_boundary(instance_mask)] = np.array([0, 255, 120], dtype=np.float32)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def compose_preview(image: Image.Image, instance_mask: np.ndarray, semantic: np.ndarray) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    colors = {
        SEMANTIC_CLEAR: np.array([0, 90, 255], dtype=np.float32),
        SEMANTIC_COMPACT: np.array([0, 150, 80], dtype=np.float32),
        SEMANTIC_STROMA: np.array([255, 230, 0], dtype=np.float32),
    }
    for semantic_value, color in colors.items():
        mask = semantic == semantic_value
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    boundaries = label_boundary(instance_mask)
    overlay[boundaries] = np.array([0, 255, 120], dtype=np.float32)
    stroma_boundary = mask_boundary(semantic == SEMANTIC_STROMA)
    overlay[stroma_boundary] = np.array([255, 255, 0], dtype=np.float32)

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def save_mask(path: Path, mask: np.ndarray) -> None:
    max_label = int(mask.max()) if mask.size else 0
    if max_label <= 255:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(mask.astype(np.uint16), mode="I;16").save(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def upsert_csv_by_tile(
    path: Path,
    fieldnames: list[str],
    tile_id: str,
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        row for row in read_existing_csv(path) if row.get("tile_id") != tile_id
    ]
    rows.extend(new_rows)
    rows.sort(key=lambda row: str(row.get("tile_id", "")))
    write_csv(path, fieldnames, rows)
    return rows


def build_datapoint(args: argparse.Namespace) -> dict[str, object]:
    output_root = Path(args.output_root).resolve()
    tile_id = args.tile_id
    size = args.image_size

    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    before_image = Image.open(before_path).convert("RGB")
    after_rgb = np.array(Image.open(after_path).convert("RGB"))

    crop = parse_crop(args.annotated_crop)
    if crop is None:
        crop = auto_content_crop(after_rgb)
    x0, y0, x1, y1 = crop
    after_crop = after_rgb[y0:y1, x0:x1, :]

    train_image = before_image.resize((size, size), Image.Resampling.BICUBIC)
    line_masks_raw = threshold_overlay_lines(after_crop)
    line_masks = {k: resize_bool(v, size) for k, v in line_masks_raw.items()}

    stroma_union, _stroma_regions = fill_closed_regions(
        line_masks["stroma"],
        min_area=args.stroma_min_area,
        max_area=args.stroma_max_area,
        line_dilate=args.stroma_line_dilate,
        close_iter=args.close_iter,
        assign_radius=args.assign_radius,
    )
    if args.stroma_line_context_dilate > 0 and line_masks["stroma"].any():
        stroma_union |= ndi.binary_dilation(
            line_masks["stroma"], iterations=args.stroma_line_context_dilate
        )
    if args.stroma_corridor_max_gap > 0:
        corridor_fill = fill_line_corridors(
            line_masks["stroma"],
            min_gap=args.stroma_corridor_min_gap,
            max_gap=args.stroma_corridor_max_gap,
            line_dilate=args.stroma_corridor_line_dilate,
            close_iter=args.stroma_corridor_close_iter,
        )
        corridor_rois = [parse_roi(value) for value in args.stroma_corridor_roi]
        if corridor_rois:
            roi_mask = np.zeros_like(corridor_fill, dtype=bool)
            for roi_x0, roi_y0, roi_x1, roi_y1 in corridor_rois:
                roi_x0 = max(roi_x0, 0)
                roi_y0 = max(roi_y0, 0)
                roi_x1 = min(roi_x1, size)
                roi_y1 = min(roi_y1, size)
                roi_mask[roi_y0:roi_y1, roi_x0:roi_x1] = True
            corridor_fill &= roi_mask
        stroma_union |= corridor_fill
    else:
        corridor_rois = []
    extra_stroma_polygons = [parse_polygon(value) for value in args.extra_stroma_polygon]
    if extra_stroma_polygons:
        stroma_union |= rasterize_polygons(extra_stroma_polygons, size)
    extra_stroma_red_masks = [
        extract_red_correction_mask(Path(value).resolve(), size)
        for value in args.extra_stroma_red_mask
    ]
    extra_stroma_red_mask_pixels = [int(mask.sum()) for mask in extra_stroma_red_masks]
    extra_stroma_red_mask_union = np.zeros((size, size), dtype=bool)
    for mask in extra_stroma_red_masks:
        extra_stroma_red_mask_union |= mask
        stroma_union |= mask
    remove_stroma_red_masks = [
        extract_orange_correction_mask(Path(value).resolve(), size)
        for value in args.remove_stroma_red_mask
    ]
    remove_stroma_red_mask_pixels = [int(mask.sum()) for mask in remove_stroma_red_masks]
    remove_stroma_red_mask_union = np.zeros((size, size), dtype=bool)
    for mask in remove_stroma_red_masks:
        remove_stroma_red_mask_union |= mask
    if remove_stroma_red_mask_union.any():
        stroma_union &= ~remove_stroma_red_mask_union
    stroma_color_trim_keep = None
    stroma_pixels_before_color_trim = int(stroma_union.sum())
    if args.stroma_color_trim:
        stroma_union, stroma_color_trim_keep = trim_stroma_to_pink_tissue(
            stroma_union,
            train_image,
            max_value=args.stroma_trim_max_value,
            min_saturation=args.stroma_trim_min_saturation,
            min_pinkness=args.stroma_trim_min_pinkness,
            dark_value=args.stroma_trim_dark_value,
            dark_min_pinkness=args.stroma_trim_dark_min_pinkness,
            close_iter=args.stroma_trim_close_iter,
            dilate=args.stroma_trim_dilate,
            min_area=args.stroma_trim_min_area,
            fill_holes_max_area=args.stroma_trim_fill_holes_max_area,
        )
    stroma_pixels_after_color_trim = int(stroma_union.sum())
    stroma_pre_postprocess = stroma_union.copy()
    if args.stroma_postprocess:
        stroma_union, stroma_postprocess_stats = postprocess_stroma_mask(
            stroma_union,
            min_region_area=args.stroma_postprocess_min_region_area,
            fill_holes_max_area=args.stroma_postprocess_fill_holes_max_area,
            closing_radius=args.stroma_postprocess_closing_radius,
            opening_radius=args.stroma_postprocess_opening_radius,
            gaussian_sigma=args.stroma_postprocess_gaussian_sigma,
            threshold=args.stroma_postprocess_threshold,
            area_change_tolerance=args.stroma_postprocess_area_change_tolerance,
            preserve_core_radius=args.stroma_postprocess_preserve_core_radius,
        )
    else:
        stroma_postprocess_stats = {
            "enabled": False,
            "pixels_before": int(stroma_union.sum()),
            "pixels_after": int(stroma_union.sum()),
            "area_change_fraction": 0.0,
            "components_before": int(ndi.label(stroma_union)[1]),
            "components_after": int(ndi.label(stroma_union)[1]),
        }
    if remove_stroma_red_mask_union.any():
        stroma_union &= ~remove_stroma_red_mask_union
        stroma_postprocess_stats["pixels_after_final_remove"] = int(stroma_union.sum())
        stroma_postprocess_stats["components_after_final_remove"] = int(
            ndi.label(stroma_union)[1]
        )
    instances_by_class: list[tuple[str, str, np.ndarray]] = []
    if args.cell_fill_combined_lines:
        cell_line_mask = line_masks["clear"] | line_masks["compact"]
        _, cell_instances = fill_closed_regions(
            cell_line_mask,
            min_area=min(args.clear_min_area, args.compact_min_area),
            max_area=max(args.clear_max_area, args.compact_max_area),
            line_dilate=args.line_dilate,
            close_iter=args.close_iter,
            assign_radius=args.assign_radius,
        )
        for mask in cell_instances:
            mask = mask & ~stroma_union
            if int(mask.sum()) < min(args.clear_min_area, args.compact_min_area):
                continue
            boundary_band = ndi.binary_dilation(
                mask_boundary(mask), iterations=args.cell_classify_boundary_dilate
            )
            compact_contact = int((boundary_band & line_masks["compact"]).sum())
            clear_contact = int((boundary_band & line_masks["clear"]).sum())
            is_compact = (
                compact_contact >= args.compact_boundary_contact_min
                and compact_contact >= clear_contact * args.compact_boundary_contact_ratio
            )
            if is_compact:
                if int(mask.sum()) >= args.compact_min_area:
                    instances_by_class.append(("compact", "GT Compact cell boundary", mask))
            elif clear_contact >= args.clear_boundary_contact_min:
                if int(mask.sum()) >= args.clear_min_area:
                    instances_by_class.append(("clear", "GT Clear cell boundary", mask))
    else:
        clear_union, clear_instances = fill_closed_regions(
            line_masks["clear"],
            min_area=args.clear_min_area,
            max_area=args.clear_max_area,
            line_dilate=args.line_dilate,
            close_iter=args.close_iter,
            assign_radius=args.assign_radius,
        )
        compact_union, compact_instances = fill_closed_regions(
            line_masks["compact"],
            min_area=args.compact_min_area,
            max_area=args.compact_max_area,
            line_dilate=args.line_dilate,
            close_iter=args.close_iter,
            assign_radius=args.assign_radius,
        )

        for mask in clear_instances:
            mask = mask & ~stroma_union
            if int(mask.sum()) >= args.clear_min_area:
                instances_by_class.append(("clear", "GT Clear cell boundary", mask))
        for mask in compact_instances:
            mask = mask & ~stroma_union & ~clear_union
            if int(mask.sum()) >= args.compact_min_area:
                instances_by_class.append(("compact", "GT Compact cell boundary", mask))

    instance_mask = np.zeros((size, size), dtype=np.uint16)
    records: list[Instance] = []
    prefix = args.object_prefix
    class_counters = {"clear": 0, "compact": 0}

    for idx, (class_key, class_name, mask) in enumerate(instances_by_class, start=1):
        if (instance_mask[mask] > 0).any():
            mask = mask & (instance_mask == 0)
        if not mask.any():
            continue
        area = int(mask.sum())
        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        class_counters[class_key] += 1
        if class_key == "clear":
            name = f"{prefix}_ClearCellBoundary_{class_counters[class_key]:03d}"
        else:
            name = f"{prefix}_CompactCellBoundary_{class_counters[class_key]:03d}"
        label = len(records) + 1
        instance_mask[mask] = label
        records.append(
            Instance(
                label=label,
                class_key=class_key,
                class_name=class_name,
                boundary_name=name,
                area_px=area,
                centroid_x=cx,
                centroid_y=cy,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
        )

    compact_binary = np.isin(instance_mask, [r.label for r in records if r.class_key == "compact"])
    clear_binary = np.isin(instance_mask, [r.label for r in records if r.class_key == "clear"])
    semantic = np.zeros((size, size), dtype=np.uint8)
    semantic[clear_binary] = SEMANTIC_CLEAR
    semantic[compact_binary] = SEMANTIC_COMPACT
    semantic[stroma_union] = SEMANTIC_STROMA

    for rel in [
        "source_pairs",
        "train/images",
        "train/masks",
        "auxiliary_masks",
        "semantic_masks",
        "previews",
        "conversion_summaries",
    ]:
        (output_root / rel).mkdir(parents=True, exist_ok=True)

    source_dir = output_root / "source_pairs" / tile_id
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(before_path, source_dir / "before.png")
    shutil.copy2(after_path, source_dir / "after_annotated.png")
    Image.fromarray(after_crop, mode="RGB").save(source_dir / "after_annotated_auto_crop.png")
    saved_extra_stroma_red_masks = []
    for idx, value in enumerate(args.extra_stroma_red_mask, start=1):
        src = Path(value).resolve()
        suffix = src.suffix or ".png"
        dest = source_dir / f"extra_stroma_red_mask_{idx:03d}{suffix}"
        shutil.copy2(src, dest)
        saved_extra_stroma_red_masks.append(str(dest.relative_to(output_root)))
    saved_remove_stroma_red_masks = []
    for idx, value in enumerate(args.remove_stroma_red_mask, start=1):
        src = Path(value).resolve()
        suffix = src.suffix or ".png"
        dest = source_dir / f"remove_stroma_red_mask_{idx:03d}{suffix}"
        shutil.copy2(src, dest)
        saved_remove_stroma_red_masks.append(str(dest.relative_to(output_root)))

    train_image_path = output_root / "train/images" / f"{tile_id}.png"
    train_mask_path = output_root / "train/masks" / f"{tile_id}.png"
    train_image.save(train_image_path)
    save_mask(train_mask_path, instance_mask)

    Image.fromarray((compact_binary.astype(np.uint8) * 255), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_compact_boundary_all.png"
    )
    Image.fromarray((clear_binary.astype(np.uint8) * 255), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_clear_boundary_all.png"
    )
    Image.fromarray((stroma_union.astype(np.uint8) * 255), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_stroma.png"
    )
    Image.fromarray((extra_stroma_red_mask_union.astype(np.uint8) * 255), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_extra_stroma_red_mask_all.png"
    )
    Image.fromarray((remove_stroma_red_mask_union.astype(np.uint8) * 255), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_remove_stroma_red_mask_all.png"
    )
    if stroma_color_trim_keep is not None:
        Image.fromarray((stroma_color_trim_keep.astype(np.uint8) * 255), mode="L").save(
            output_root / "auxiliary_masks" / f"{tile_id}_stroma_color_trim_keep.png"
        )
    if args.stroma_postprocess:
        Image.fromarray(
            (stroma_pre_postprocess.astype(np.uint8) * 255), mode="L"
        ).save(output_root / "auxiliary_masks" / f"{tile_id}_gt_stroma_pre_postprocess.png")
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_uncertain_ignore.png"
    )
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_edge_or_invalid_cell_ignore.png"
    )
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_nucleus_instances.png"
    )
    Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L").save(
        output_root / "auxiliary_masks" / f"{tile_id}_gt_nucleus_all_direct_children.png"
    )
    Image.fromarray(semantic, mode="L").save(
        output_root / "semantic_masks" / f"{tile_id}_semantic_review.png"
    )
    compose_preview(train_image, instance_mask, semantic).save(
        output_root / "previews" / f"{tile_id}_overlay.png"
    )
    compose_instance_label_preview(train_image, instance_mask).save(
        output_root / "previews" / f"{tile_id}_instance_labels.png"
    )

    instance_rows = []
    for record in records:
        instance_rows.append(
            {
                "instance_label": record.label,
                "boundary_name": record.boundary_name,
                "boundary_class": record.class_name,
                "nucleus_name": "",
                "boundary_area_px": f"{record.area_px:.1f}",
                "local_centroid_x": f"{record.centroid_x:.1f}",
                "local_centroid_y": f"{record.centroid_y:.1f}",
                "wsi_centroid_x": "",
                "wsi_centroid_y": "",
                "nucleus_wsi_x": "",
                "nucleus_wsi_y": "",
                "stroma_overlap_px": "0.0",
                "stroma_overlap_fraction": "0.0000",
            }
        )

    instance_fields = [
        "instance_label",
        "boundary_name",
        "boundary_class",
        "nucleus_name",
        "boundary_area_px",
        "local_centroid_x",
        "local_centroid_y",
        "wsi_centroid_x",
        "wsi_centroid_y",
        "nucleus_wsi_x",
        "nucleus_wsi_y",
        "stroma_overlap_px",
        "stroma_overlap_fraction",
    ]
    write_csv(output_root / f"{tile_id}_instances.csv", instance_fields, instance_rows)

    cell_rows = []
    for row in instance_rows:
        merged = {"tile_id": tile_id, "tile_name": args.tile_name}
        merged.update(row)
        cell_rows.append(merged)
    all_cell_rows = upsert_csv_by_tile(
        output_root / "cell_instances.csv",
        ["tile_id", "tile_name"] + instance_fields,
        tile_id,
        cell_rows,
    )

    qc_rows = []
    for record in records:
        qc_rows.append(
            {
                "tile_id": tile_id,
                "tile_name": args.tile_name,
                "boundary_name": record.boundary_name,
                "boundary_class": record.class_name,
                "include_for_cellseg1": "true",
                "nuclei_inside": "NA",
                "nuclei_inside_in_tile": "NA",
                "nuclei_names": "",
                "stroma_overlap_px": "0.0",
                "stroma_overlap_fraction": "0.0000",
                "flags": "PNG_DERIVED_NO_NUCLEUS_QC",
            }
        )
    all_qc_rows = upsert_csv_by_tile(
        output_root / "boundary_qc.csv",
        [
            "tile_id",
            "tile_name",
            "boundary_name",
            "boundary_class",
            "include_for_cellseg1",
            "nuclei_inside",
            "nuclei_inside_in_tile",
            "nuclei_names",
            "stroma_overlap_px",
            "stroma_overlap_fraction",
            "flags",
        ],
        tile_id,
        qc_rows,
    )

    manifest_row = {
        "tile_id": tile_id,
        "tile_name": args.tile_name,
        "image_file": f"train/images/{tile_id}.png",
        "mask_file": f"train/masks/{tile_id}.png",
        "width": size,
        "height": size,
        "trainable_instances": len(records),
        "clear_trainable": class_counters["clear"],
        "compact_trainable": class_counters["compact"],
        "edge_or_invalid_boundary_ignore": 0,
        "uncertain_ignore_regions": 0,
        "nuclei_total": 0,
        "nuclei_in_tile": 0,
        "stroma_regions_intersecting_tile": int(ndi.label(stroma_union)[1]),
    }
    all_manifest_rows = upsert_csv_by_tile(
        output_root / "dataset_manifest.csv",
        list(manifest_row.keys()),
        tile_id,
        [manifest_row],
    )

    summary = {
        "tile_id": tile_id,
        "source_type": "png_derived_from_human_overlay",
        "before_image": str(before_path),
        "after_annotated_image": str(after_path),
        "annotated_crop_xyxy": crop,
        "output_image_size": [size, size],
        "trainable_instances": len(records),
        "clear_trainable": class_counters["clear"],
        "compact_trainable": class_counters["compact"],
        "stroma_regions_intersecting_tile": int(ndi.label(stroma_union)[1]),
        "line_pixels_512": {key: int(value.sum()) for key, value in line_masks.items()},
        "cell_fill": {
            "mode": "combined_lines" if args.cell_fill_combined_lines else "separate_lines",
            "line_dilate": args.line_dilate,
            "classify_boundary_dilate": args.cell_classify_boundary_dilate,
            "compact_boundary_contact_min": args.compact_boundary_contact_min,
            "compact_boundary_contact_ratio": args.compact_boundary_contact_ratio,
            "clear_boundary_contact_min": args.clear_boundary_contact_min,
        },
        "stroma_corridor_fill": {
            "min_gap": args.stroma_corridor_min_gap,
            "max_gap": args.stroma_corridor_max_gap,
            "line_dilate": args.stroma_corridor_line_dilate,
            "close_iter": args.stroma_corridor_close_iter,
            "rois_512": corridor_rois,
        },
        "stroma_color_trim": {
            "enabled": bool(args.stroma_color_trim),
            "pixels_before": stroma_pixels_before_color_trim,
            "pixels_after": stroma_pixels_after_color_trim,
            "max_value": args.stroma_trim_max_value,
            "min_saturation": args.stroma_trim_min_saturation,
            "min_pinkness": args.stroma_trim_min_pinkness,
            "dark_value": args.stroma_trim_dark_value,
            "dark_min_pinkness": args.stroma_trim_dark_min_pinkness,
            "close_iter": args.stroma_trim_close_iter,
            "dilate": args.stroma_trim_dilate,
            "min_area": args.stroma_trim_min_area,
            "fill_holes_max_area": args.stroma_trim_fill_holes_max_area,
        },
        "stroma_postprocess": stroma_postprocess_stats,
        "extra_stroma_polygons_512": extra_stroma_polygons,
        "extra_stroma_red_masks": [
            {
                "source_path": str(Path(value).resolve()),
                "copied_file": copied_file,
                "pixels_512": pixel_count,
            }
            for value, copied_file, pixel_count in zip(
                args.extra_stroma_red_mask,
                saved_extra_stroma_red_masks,
                extra_stroma_red_mask_pixels,
            )
        ],
        "remove_stroma_red_masks": [
            {
                "source_path": str(Path(value).resolve()),
                "copied_file": copied_file,
                "pixels_512": pixel_count,
            }
            for value, copied_file, pixel_count in zip(
                args.remove_stroma_red_mask,
                saved_remove_stroma_red_masks,
                remove_stroma_red_mask_pixels,
            )
        ],
        "notes": [
            "Derived from colored overlay screenshot, not original QuPath vector objects.",
            "Nucleus QC is unavailable; boundary_qc.csv marks PNG_DERIVED_NO_NUCLEUS_QC.",
            "Only closed filled regions are exported as trainable positive instances.",
        ],
    }
    with (output_root / "conversion_summaries" / f"{tile_id}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    summary_files = sorted(
        p
        for p in (output_root / "conversion_summaries").glob("*.json")
        if not p.name.startswith("._")
    )
    summaries = []
    for summary_file in summary_files:
        with summary_file.open(encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    batch_summary = {
        "source_type": "png_derived_from_human_overlay_batch",
        "datapoints": summaries,
        "tile_count": len(all_manifest_rows),
        "trainable_instances": sum(int(row["trainable_instances"]) for row in all_manifest_rows),
        "clear_trainable": sum(int(row["clear_trainable"]) for row in all_manifest_rows),
        "compact_trainable": sum(int(row["compact_trainable"]) for row in all_manifest_rows),
        "cell_instances_rows": len(all_cell_rows),
        "boundary_qc_rows": len(all_qc_rows),
    }
    with (output_root / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(batch_summary, handle, indent=2)

    manifest_lines = "\n".join(
        f"- `{row['tile_id']}`: {row['trainable_instances']} trainable cell instances "
        f"({row['clear_trainable']} clear, {row['compact_trainable']} compact), "
        f"{row['stroma_regions_intersecting_tile']} stroma regions."
        for row in all_manifest_rows
    )

    readme = f"""# CGH P2 CellSeg1 Second Batch

This folder contains PNG-derived training datapoints reconstructed from human
before/after annotation screenshots.

Current datapoints: {len(all_manifest_rows)}

{manifest_lines}

Important caveats:

- This is not a canonical QuPath vector export.
- The instance mask is reconstructed from colored overlay lines, so review
  `previews/{tile_id}_overlay.png` before training.
- Nucleus objects are not available from the screenshot; nucleus QC fields are
  marked `PNG_DERIVED_NO_NUCLEUS_QC`.
- Use this batch as supplemental data or convert future human annotation pairs
  with `tools/create_png_derived_datapoint.py`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Raw/before tile image")
    parser.add_argument("--after", required=True, help="After image with colored overlays")
    parser.add_argument("--output-root", required=True, help="Second-batch output folder")
    parser.add_argument("--tile-id", default="human_compact_tile_001")
    parser.add_argument("--tile-name", default="Human compact tile 001")
    parser.add_argument("--object-prefix", default="HCT001")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--annotated-crop", default="auto", help="auto or x0,y0,x1,y1")
    parser.add_argument("--line-dilate", type=int, default=2)
    parser.add_argument("--stroma-line-dilate", type=int, default=4)
    parser.add_argument("--stroma-line-context-dilate", type=int, default=2)
    parser.add_argument("--stroma-corridor-min-gap", type=int, default=4)
    parser.add_argument("--stroma-corridor-max-gap", type=int, default=0)
    parser.add_argument("--stroma-corridor-line-dilate", type=int, default=2)
    parser.add_argument("--stroma-corridor-close-iter", type=int, default=2)
    parser.add_argument(
        "--stroma-corridor-roi",
        action="append",
        default=[],
        help="Limit corridor fill to x0,y0,x1,y1 in 512x512 output coords.",
    )
    parser.add_argument(
        "--stroma-color-trim",
        action="store_true",
        help="Trim stroma to darker pink tissue in the raw tile, excluding pale/white cells.",
    )
    parser.add_argument("--stroma-trim-max-value", type=float, default=238.0)
    parser.add_argument("--stroma-trim-min-saturation", type=float, default=0.075)
    parser.add_argument("--stroma-trim-min-pinkness", type=float, default=15.0)
    parser.add_argument("--stroma-trim-dark-value", type=float, default=210.0)
    parser.add_argument("--stroma-trim-dark-min-pinkness", type=float, default=8.0)
    parser.add_argument("--stroma-trim-close-iter", type=int, default=1)
    parser.add_argument("--stroma-trim-dilate", type=int, default=1)
    parser.add_argument("--stroma-trim-min-area", type=int, default=8)
    parser.add_argument("--stroma-trim-fill-holes-max-area", type=int, default=0)
    parser.add_argument(
        "--stroma-postprocess",
        action="store_true",
        help="Smooth and clean the final binary stroma mask.",
    )
    parser.add_argument("--stroma-postprocess-min-region-area", type=int, default=24)
    parser.add_argument("--stroma-postprocess-fill-holes-max-area", type=int, default=96)
    parser.add_argument("--stroma-postprocess-closing-radius", type=int, default=1)
    parser.add_argument("--stroma-postprocess-opening-radius", type=int, default=1)
    parser.add_argument("--stroma-postprocess-gaussian-sigma", type=float, default=0.8)
    parser.add_argument("--stroma-postprocess-threshold", type=float, default=0.50)
    parser.add_argument("--stroma-postprocess-area-change-tolerance", type=float, default=0.08)
    parser.add_argument("--stroma-postprocess-preserve-core-radius", type=int, default=1)
    parser.add_argument(
        "--extra-stroma-polygon",
        action="append",
        default=[],
        help="Extra filled stroma polygon in 512x512 output coords: 'x,y x,y ...'",
    )
    parser.add_argument(
        "--extra-stroma-red-mask",
        action="append",
        default=[],
        help="Preview screenshot with red user-highlighted stroma gaps to add.",
    )
    parser.add_argument(
        "--remove-stroma-red-mask",
        action="append",
        default=[],
        help="Preview screenshot with red/orange user-highlighted false stroma to remove.",
    )
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--assign-radius", type=float, default=4.0)
    parser.add_argument("--compact-min-area", type=int, default=80)
    parser.add_argument("--compact-max-area", type=int, default=6500)
    parser.add_argument("--clear-min-area", type=int, default=120)
    parser.add_argument("--clear-max-area", type=int, default=9000)
    parser.add_argument(
        "--cell-fill-combined-lines",
        action="store_true",
        help="Fill cells from clear+compact line union, then classify by boundary color contact.",
    )
    parser.add_argument("--cell-classify-boundary-dilate", type=int, default=4)
    parser.add_argument("--compact-boundary-contact-min", type=int, default=20)
    parser.add_argument("--compact-boundary-contact-ratio", type=float, default=0.25)
    parser.add_argument("--clear-boundary-contact-min", type=int, default=20)
    parser.add_argument("--stroma-min-area", type=int, default=120)
    parser.add_argument("--stroma-max-area", type=int, default=40000)
    args = parser.parse_args()

    summary = build_datapoint(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
