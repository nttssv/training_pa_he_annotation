"""Mask geometry helpers for nucleus-boundary matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_mask(path: Path | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if path is None or not path.exists():
        if shape is None:
            return np.zeros((0, 0), dtype=np.uint16)
        return np.zeros(shape, dtype=np.uint16)
    return np.asarray(Image.open(path))


def instance_labels(mask: np.ndarray) -> list[int]:
    if mask.size == 0:
        return []
    return sorted(int(value) for value in np.unique(mask) if int(value) != 0)


def instance_area(mask: np.ndarray, label: int) -> int:
    return int(np.count_nonzero(mask == label))


def binary_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def bbox_for_binary(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def centroid_for_binary(binary: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroids_from_instance_mask(mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in instance_labels(mask):
        binary = mask == label
        centroid = centroid_for_binary(binary)
        bbox = bbox_for_binary(binary)
        if centroid is None or bbox is None:
            continue
        rows.append(
            {
                "label": label,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "area_px": instance_area(mask, label),
                "bbox": bbox,
            }
        )
    return rows


def point_inside_mask(mask: np.ndarray, x: float, y: float) -> bool:
    if mask.size == 0:
        return False
    xi = int(round(x))
    yi = int(round(y))
    if yi < 0 or xi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1]:
        return False
    return bool(mask[yi, xi])


def touches_edge(binary: np.ndarray, margin_px: int) -> bool:
    if binary.size == 0 or not np.any(binary):
        return False
    margin = max(int(margin_px), 0)
    if margin == 0:
        return bool(binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any())
    return bool(
        binary[: margin + 1, :].any()
        or binary[-margin - 1 :, :].any()
        or binary[:, : margin + 1].any()
        or binary[:, -margin - 1 :].any()
    )


def overlap_fraction(binary: np.ndarray, other: np.ndarray) -> float:
    if binary.size == 0 or other.size == 0 or not np.any(binary):
        return 0.0
    if binary.shape != other.shape:
        return 0.0
    return float(np.count_nonzero(binary & (other > 0)) / max(np.count_nonzero(binary), 1))


def mask_boundary(binary: np.ndarray) -> np.ndarray:
    mask = binary.astype(bool)
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    return boundary & mask

