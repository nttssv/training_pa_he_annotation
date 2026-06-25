"""Progress, ETA, runtime summary, and failure-report helpers."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from training.utils.reproducibility import cuda_info, git_commit_hash


class ProgressTracker:
    def __init__(self, stage: str, total_epochs: int = 1, total_batches: int = 1):
        self.stage = stage
        self.total_epochs = max(int(total_epochs), 1)
        self.total_batches = max(int(total_batches), 1)
        self.start = time.time()
        self.current_epoch = 0
        self.current_batch = 0

    def update(self, epoch: int, batch: int) -> dict[str, Any]:
        self.current_epoch = int(epoch)
        self.current_batch = int(batch)
        elapsed = time.time() - self.start
        units_done = max((self.current_epoch - 1) * self.total_batches + self.current_batch, 1)
        total_units = self.total_epochs * self.total_batches
        seconds_per_unit = elapsed / units_done
        seconds_per_epoch = seconds_per_unit * self.total_batches
        remaining = max(total_units - units_done, 0) * seconds_per_unit
        return {
            "start_time": datetime.fromtimestamp(self.start).isoformat(timespec="seconds"),
            "current_time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "time_per_batch_seconds": seconds_per_unit,
            "time_per_epoch_seconds": seconds_per_epoch,
            "estimated_remaining_seconds": remaining,
            "estimated_finish_time": (datetime.now() + timedelta(seconds=remaining)).isoformat(timespec="seconds"),
            "current_stage": self.stage,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
        }

    def write_runtime_summary(self, path: Path, epoch: int, batch: int) -> dict[str, Any]:
        summary = self.update(epoch, batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def format_seconds(seconds: float) -> str:
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def system_usage() -> dict[str, Any]:
    usage: dict[str, Any] = {"ram_used_gb": "", "ram_total_gb": "", "gpu_name": "", "gpu_memory_gb": ""}
    try:
        import psutil

        mem = psutil.virtual_memory()
        usage["ram_used_gb"] = round((mem.total - mem.available) / 1024**3, 2)
        usage["ram_total_gb"] = round(mem.total / 1024**3, 2)
    except Exception:
        pass
    info = cuda_info()
    if info.get("devices"):
        usage["gpu_name"] = info["devices"][0]
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(idx) / 1024**3
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            usage["gpu_memory_gb"] = f"{used:.2f}/{total:.2f}"
    except Exception:
        pass
    return usage


def print_epoch_progress(
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    runtime: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_path: str | Path,
) -> None:
    usage = system_usage()
    print(
        f"Epoch {epoch:03d}/{total_epochs:03d} | Batch {batch:03d}/{total_batches:03d} | "
        f"ETA {format_seconds(float(runtime.get('estimated_remaining_seconds', 0)))}"
    )
    print(
        f"GPU {usage.get('gpu_name') or 'unavailable'} | VRAM {usage.get('gpu_memory_gb') or 'unavailable'} GB | "
        f"RAM {usage.get('ram_used_gb') or '?'} / {usage.get('ram_total_gb') or '?'} GB"
    )
    print(
        "lr={learning_rate} | train_loss={train_loss} | val_loss={val_loss} | "
        "val_dice={val_dice} | val_iou={val_iou} | best_val_dice={best_val_dice} | checkpoint={checkpoint}".format(
            learning_rate=metrics.get("learning_rate", ""),
            train_loss=metrics.get("train_loss", ""),
            val_loss=metrics.get("val_loss", ""),
            val_dice=metrics.get("val_dice", ""),
            val_iou=metrics.get("val_iou", ""),
            best_val_dice=metrics.get("best_val_dice", ""),
            checkpoint=checkpoint_path,
        )
    )


def write_run_summary(path: Path, cfg: dict[str, Any], dataset_summary: dict[str, Any], output_folder: Path) -> None:
    cuda = cuda_info()
    lines = [
        "Adrenal H&E morphology run summary",
        f"experiment: {cfg.get('experiment', {}).get('name', '')}",
        f"dataset path: {cfg.get('paths', {}).get('dataset_root', '')}",
        f"train images: {dataset_summary.get('train_images', '')}",
        f"validation images: {dataset_summary.get('val_images', '')}",
        f"classes: {dataset_summary.get('classes', '')}",
        f"batch size: {cfg.get('models', {}).get('boundary', {}).get('batch_size', '')}",
        f"epochs: {cfg.get('models', {}).get('boundary', {}).get('epochs', '')}",
        f"image size: {cfg.get('models', {}).get('boundary', {}).get('resize_size', '')}",
        f"model type: YOLO nuclei + CellSeg1/SAM boundary",
        f"resume checkpoint: {cfg.get('runtime', {}).get('resume_checkpoint', '')}",
        f"device: {cfg.get('runtime', {}).get('device', '')}",
        f"CUDA info: {cuda}",
        f"random seed: {cfg.get('experiment', {}).get('seed', '')}",
        f"output folder: {output_folder}",
        f"git commit hash: {git_commit_hash()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_report(
    path: Path,
    exc: BaseException,
    stage: str,
    epoch: int | None = None,
    batch: int | None = None,
    last_checkpoint: str | Path | None = None,
) -> None:
    message = str(exc)
    lower = message.lower()
    if "out of memory" in lower or "cuda" in lower and "memory" in lower:
        likely = "CUDA OOM"
    elif "no such file" in lower or "does not exist" in lower or "missing" in lower:
        likely = "missing file/path"
    elif "size mismatch" in lower or "mismatch" in lower:
        likely = "annotation or image/mask size mismatch"
    elif "nan" in lower:
        likely = "NaN loss"
    elif "polygon" in lower:
        likely = "invalid polygon"
    else:
        likely = "unknown; inspect stack trace and previous log lines"
    text = [
        "Adrenal H&E morphology pipeline error report",
        f"stage: {stage}",
        f"epoch: {epoch if epoch is not None else ''}",
        f"batch: {batch if batch is not None else ''}",
        f"last checkpoint: {last_checkpoint or ''}",
        f"likely cause: {likely}",
        "",
        "error message:",
        message,
        "",
        "stack trace:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")
