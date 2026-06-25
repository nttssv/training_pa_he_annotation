"""Train or smoke-test the CellSeg1/SAM-style boundary model."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from training.split_dataset import split_pairs
from training.train_yolo_nuclei import smoke_forward_backward
from training.utils.io import PipelineError, dataset_paths, resolve_path, write_csv, write_json
from training.utils.progress import ProgressTracker, print_epoch_progress
from training.utils.visualization import overlay_masks, plot_history


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_dice",
    "val_iou",
    "learning_rate",
    "elapsed_seconds",
    "eta_seconds",
    "gpu_memory_gb",
    "valid_cell_rate",
    "uncertain_cell_rate",
]


def write_history_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})


def smoke(cfg: dict[str, Any], dirs: dict[str, Path], visualize: bool = False) -> dict[str, Any]:
    train, val = split_pairs(cfg)
    limit = int(cfg.get("smoke", {}).get("num_images", 3))
    sample = (train + val)[:limit]
    if not sample:
        raise PipelineError("dataset", "Smoke test found no image/mask pairs")

    summary = smoke_forward_backward(cfg, dirs)
    history = dirs["metrics"] / "training_history.csv"
    tracker = ProgressTracker("smoke", 1, 1)
    runtime = tracker.write_runtime_summary(dirs["logs"] / "runtime_summary.json", 1, 1)
    loss_value = summary.get("loss")
    try:
        loss_for_history = float(loss_value)
    except Exception:
        loss_for_history = ""
    write_history_row(
        history,
        {
            "epoch": 1,
            "train_loss": loss_for_history,
            "val_loss": "",
            "val_dice": "",
            "val_iou": "",
            "learning_rate": 1e-3,
            "elapsed_seconds": runtime["elapsed_seconds"],
            "eta_seconds": runtime["estimated_remaining_seconds"],
            "gpu_memory_gb": "",
            "valid_cell_rate": "",
            "uncertain_cell_rate": "",
        },
    )
    plot_history(history, dirs["run"] / "plots")
    write_tensorboard_smoke(dirs, summary)

    if visualize:
        epoch_dir = dirs["overlays"] / "epoch_001"
        for row in sample[:2]:
            overlay_masks(
                row["image_path"],
                epoch_dir / f"{row['tile_id']}_ground_truth_overlay.png",
                boundary_mask=row["mask_path"],
                nucleus_mask=row["nucleus_mask_path"],
                uncertain_mask=row["uncertain_mask_path"],
                edge_mask=row["edge_mask_path"],
                title=f"{row['tile_id']} GT smoke",
            )
            overlay_masks(
                row["image_path"],
                epoch_dir / f"{row['tile_id']}_prediction_overlay.png",
                boundary_mask=row["mask_path"],
                nucleus_mask=row["nucleus_mask_path"],
                title=f"{row['tile_id']} placeholder prediction",
            )
            overlay_masks(
                row["image_path"],
                epoch_dir / f"{row['tile_id']}_error_overlay.png",
                uncertain_mask=row["uncertain_mask_path"],
                edge_mask=row["edge_mask_path"],
                title=f"{row['tile_id']} QC/error regions",
            )
    write_json(dirs["metrics"] / "smoke_summary.json", summary)
    return summary


def write_tensorboard_smoke(dirs: dict[str, Path], summary: dict[str, Any]) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return
    writer = SummaryWriter(log_dir=str(dirs["run"] / "tensorboard"))
    try:
        writer.add_scalar("train/loss", float(summary.get("loss", 0.0)), 1)
    except Exception:
        pass
    writer.add_scalar("runtime/smoke_complete", 1, 1)
    writer.flush()
    writer.close()


def build_cellseg1_config(cfg: dict[str, Any], dirs: dict[str, Path]) -> Path:
    boundary_cfg = cfg.get("models", {}).get("boundary", {})
    resolved_dataset = dataset_paths(cfg)
    dataset_root = resolved_dataset["root"]
    checkpoint = boundary_cfg.get("sam_checkpoint", "")
    if not checkpoint:
        raise PipelineError("config", "models.boundary.sam_checkpoint is required for CellSeg1 training")
    checkpoint_path = resolve_path(cfg, checkpoint)
    if not checkpoint_path.exists():
        raise PipelineError("path", f"SAM checkpoint not found: {checkpoint_path}")

    config = {
        "deterministic": True,
        "allow_tf32_on_matmul": True,
        "allow_tf32_on_cudnn": True,
        "seed": int(cfg.get("experiment", {}).get("seed", 42)),
        "vit_name": boundary_cfg.get("vit_name", "vit_h"),
        "model_path": str(checkpoint_path),
        "data_dir": str(dataset_root),
        "result_pth_path": str(dirs["checkpoints"] / "cell_boundary_lora.pth"),
        "train_image_dir": str(resolved_dataset["images"]),
        "train_mask_dir": str(resolved_dataset["boundary_masks"]),
        "resize_size": boundary_cfg.get("resize_size", [512, 512]),
        "patch_size": 0,
        "sam_image_size": int(boundary_cfg.get("sam_image_size", 512)),
        "train_id": boundary_cfg.get("train_id"),
        "duplicate_data": int(boundary_cfg.get("duplicate_data", 64)),
        "epoch_max": int(boundary_cfg.get("epochs", 120)),
        "batch_size": int(boundary_cfg.get("batch_size", 1)),
        "gradient_accumulation_step": int(boundary_cfg.get("gradient_accumulation_step", 32)),
        "base_lr": float(boundary_cfg.get("learning_rate", 0.003)),
        "num_workers": int(boundary_cfg.get("workers", 0)),
        "image_encoder_lora_rank": int(boundary_cfg.get("image_encoder_lora_rank", 4)),
        "mask_decoder_lora_rank": int(boundary_cfg.get("mask_decoder_lora_rank", 4)),
        "freeze_image_encoder": True,
        "freeze_prompt_encoder": True,
        "freeze_mask_decoder_transformer": True,
        "freeze_upscaling_cnn": True,
        "freeze_output_hypernetworks_mlps": True,
        "freeze_mask_decoder_mask_tokens": True,
        "freeze_mask_decoder_iou": True,
        "lora_dropout": float(boundary_cfg.get("lora_dropout", 0.1)),
        "pos_rate": 1.0,
        "neg_rate": 0.0,
        "max_point_num": 30,
        "edge_distance": 20,
        "min_cell_area": int(cfg.get("qc", {}).get("min_cell_area_px", 100)),
        "data_augmentation": True,
        "bright_limit": 0.1,
        "contrast_limit": 0.1,
        "bright_prob": 0.5,
        "flip_prob": 0.75,
        "rotate_prob": 0.8,
        "scale_limit": [-0.5, 0.5],
        "crop_prob": 0.5,
        "crop_scale": [0.3, 1.0],
        "crop_ratio": [0.75, 1.3333],
        "ce_loss_weight": 1.0,
        "punish_background_point": False,
    }
    path = dirs["configs"] / "cellseg1_boundary_runtime.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def train(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    boundary_cfg = cfg.get("models", {}).get("boundary", {})
    repo = resolve_path(cfg, boundary_cfg.get("cellseg1_repo", "qupath_work/cellseg1_repo"))
    runner = Path(__file__).resolve().parent / "run_cellseg1_lora_train.py"
    if not repo.exists():
        raise PipelineError("path", f"CellSeg1 repo not found: {repo}")
    if not runner.exists():
        raise PipelineError("path", f"CellSeg1 runner not found: {runner}")
    config_path = build_cellseg1_config(cfg, dirs)
    summary_path = dirs["metrics"] / "cell_boundary_training_summary.json"
    log_path = dirs["logs"] / "train_cell_boundary.log"
    cmd = [sys.executable, str(runner), "--repo", str(repo), "--config", str(config_path), "--summary", str(summary_path)]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", "offline")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise PipelineError("training", f"CellSeg1 boundary training failed with return code {proc.returncode}. See {log_path}")
    summary = {"command": " ".join(str(x) for x in cmd), "log": str(log_path), "summary": str(summary_path)}
    write_json(dirs["metrics"] / "train_boundary_summary.json", summary)
    return summary


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return train(cfg, dirs)
