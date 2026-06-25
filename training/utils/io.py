"""IO, configuration, and dataset-discovery helpers."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


class PipelineError(RuntimeError):
    """Error with a category prefix intended for cluster logs."""

    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(f"[{category}] {message}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PipelineError("config", f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise PipelineError("config", f"Config must be a YAML mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    cfg = expand_env_values(read_yaml(path))
    cfg["_config_path"] = str(path)

    thresholds_path = cfg.get("thresholds_config")
    if thresholds_path:
        threshold_file = resolve_path({"paths": {}}, thresholds_path, base=repo_root())
        thresholds = expand_env_values(read_yaml(threshold_file))
        cfg = deep_merge(thresholds, cfg)
        cfg["_thresholds_config_path"] = str(threshold_file)
    return cfg


def require_config(cfg: dict[str, Any], dotted_key: str) -> Any:
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PipelineError("config", f"Missing required config key: {dotted_key}")
        current = current[part]
    return current


def resolve_path(cfg: dict[str, Any], value: str | Path, base: Path | None = None) -> Path:
    raw = expand_env_string(str(value))
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_string(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        current = os.environ.get(name)
        if current not in (None, ""):
            return current
        return default or ""

    return os.path.expandvars(ENV_DEFAULT_RE.sub(repl, value))


def expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_string(value)
    if isinstance(value, list):
        return [expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    return value


def dataset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = resolve_path(cfg, require_config(cfg, "paths.dataset_root"))
    return infer_dataset_paths(
        root=root,
        image_dir=require_config(cfg, "paths.image_dir"),
        boundary_mask_dir=require_config(cfg, "paths.boundary_mask_dir"),
        auxiliary_mask_dir=require_config(cfg, "paths.auxiliary_mask_dir"),
        metadata_dir=require_config(cfg, "paths.metadata_dir"),
    )


def infer_dataset_paths(
    root: Path,
    image_dir: str,
    boundary_mask_dir: str,
    auxiliary_mask_dir: str,
    metadata_dir: str,
) -> dict[str, Path]:
    """Resolve dataset folders and tolerate common packaged layouts.

    Supported layouts include:
    - root/images + root/masks
    - root/train/images + root/train/masks
    - root/dataset/images + root/dataset/masks
    """
    images = root / image_dir
    boundary_masks = root / boundary_mask_dir
    auxiliary_masks = root / auxiliary_mask_dir
    metadata = root / metadata_dir

    candidates = [
        {
            "images": root / "images",
            "boundary_masks": root / "masks",
            "auxiliary_masks": root / "auxiliary_masks",
            "metadata": root / "metadata",
        },
        {
            "images": root / "train" / "images",
            "boundary_masks": root / "train" / "masks",
            "auxiliary_masks": root / "auxiliary_masks",
            "metadata": root,
        },
        {
            "images": root / "dataset" / "images",
            "boundary_masks": root / "dataset" / "masks",
            "auxiliary_masks": root / "dataset" / "auxiliary_masks",
            "metadata": root / "dataset" / "metadata",
        },
    ]
    if not images.exists() or not boundary_masks.exists():
        for candidate in candidates:
            if candidate["images"].exists() and candidate["boundary_masks"].exists():
                images = candidate["images"]
                boundary_masks = candidate["boundary_masks"]
                if candidate["auxiliary_masks"].exists():
                    auxiliary_masks = candidate["auxiliary_masks"]
                if candidate["metadata"].exists():
                    metadata = candidate["metadata"]
                break
    if not auxiliary_masks.exists():
        for candidate in candidates:
            if candidate["auxiliary_masks"].exists():
                auxiliary_masks = candidate["auxiliary_masks"]
                break
    if not metadata.exists():
        for candidate in candidates:
            if candidate["metadata"].exists():
                metadata = candidate["metadata"]
                break
    return {
        "root": root,
        "images": images,
        "boundary_masks": boundary_masks,
        "auxiliary_masks": auxiliary_masks,
        "metadata": metadata,
    }


def ensure_dataset_layout(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = dataset_paths(cfg)
    missing = [name for name, path in paths.items() if name != "metadata" and not path.exists()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        hint = (
            "Expected either root/images + root/masks or root/train/images + root/train/masks. "
            "Check with: find \"$CGH_DATASET_ROOT\" -maxdepth 3 -type d | sort"
        )
        raise PipelineError("path", f"Dataset layout is incomplete: {detail}. {hint}")
    return paths


def image_extensions(cfg: dict[str, Any]) -> tuple[str, ...]:
    values = cfg.get("data", {}).get("image_extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    return tuple(str(v).lower() for v in values)


def list_images(image_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not image_dir.exists():
        raise PipelineError("path", f"Image directory does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in extensions
    )


def matching_mask_for_image(mask_dir: Path, image_path: Path) -> Path | None:
    for suffix in (image_path.suffix, ".png", ".tif", ".tiff"):
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists() and not candidate.name.startswith("._"):
            return candidate
    return None


def load_manifest_metadata(metadata_dir: Path) -> dict[str, dict[str, str]]:
    manifest = metadata_dir / "dataset_manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("tile_id", row.get("image_id", "")).strip(): row for row in rows if row}


def load_cell_class_map(metadata_dir: Path) -> dict[tuple[str, int], str]:
    path = metadata_dir / "cell_instances.csv"
    if not path.exists():
        return {}
    mapping: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tile_id = (row.get("tile_id") or "").strip()
            raw_label = (row.get("instance_label") or "").strip()
            if not tile_id or not raw_label:
                continue
            try:
                label = int(float(raw_label))
            except ValueError:
                continue
            mapping[(tile_id, label)] = (row.get("boundary_class") or "").strip()
    return mapping


def discover_pairs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    paths = ensure_dataset_layout(cfg)
    images = list_images(paths["images"], image_extensions(cfg))
    metadata = load_manifest_metadata(paths["metadata"])
    pairs: list[dict[str, Any]] = []
    for image_path in images:
        mask_path = matching_mask_for_image(paths["boundary_masks"], image_path)
        tile_id = image_path.stem
        row = metadata.get(tile_id, {})
        batch_source = row.get("batch_source") or row.get("source_batch") or infer_batch_source(tile_id)
        case_id = row.get("case_id") or row.get("slide_id") or ""
        nucleus_mask = paths["auxiliary_masks"] / f"{tile_id}_gt_nucleus_instances.png"
        stroma_mask = paths["auxiliary_masks"] / f"{tile_id}_gt_stroma.png"
        uncertain_mask = paths["auxiliary_masks"] / f"{tile_id}_gt_uncertain_ignore.png"
        edge_mask = paths["auxiliary_masks"] / f"{tile_id}_edge_or_invalid_cell_ignore.png"
        pairs.append(
            {
                "tile_id": tile_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "nucleus_mask_path": nucleus_mask if nucleus_mask.exists() else None,
                "stroma_mask_path": stroma_mask if stroma_mask.exists() else None,
                "uncertain_mask_path": uncertain_mask if uncertain_mask.exists() else None,
                "edge_mask_path": edge_mask if edge_mask.exists() else None,
                "batch_source": batch_source,
                "case_id": case_id,
            }
        )
    return pairs


def infer_batch_source(tile_id: str) -> str:
    if "_" in tile_id:
        return tile_id.rsplit("_", 1)[0]
    return ""


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def class_distribution_from_metadata(metadata_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    path = metadata_dir / "cell_instances.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                klass = (row.get("boundary_class") or "unknown").strip() or "unknown"
                counts[klass] += 1
    return dict(counts)
