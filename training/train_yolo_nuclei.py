"""Train or smoke-test the YOLO nuclei model."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from training.utils.io import PipelineError, resolve_path, write_json
from training.utils.progress import ProgressTracker, print_epoch_progress
from training.utils.visualization import plot_history


def _require_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception as exc:
        raise PipelineError(
            "dependency",
            "Ultralytics is required for YOLO nuclei training. Install with: python -m pip install ultralytics",
        ) from exc


def train(cfg: dict[str, Any], dirs: dict[str, Path], smoke: bool = False) -> dict[str, Any]:
    _require_ultralytics()
    yolo_cfg = cfg.get("models", {}).get("nuclei", {})
    data_yaml = resolve_path(cfg, yolo_cfg.get("data_yaml", "training_data/dataset/yolo_seg_dataset/data.yaml"))
    if not data_yaml.exists():
        raise PipelineError("path", f"YOLO data.yaml not found: {data_yaml}")
    epochs = int(yolo_cfg.get("smoke_epochs" if smoke else "epochs", 1 if smoke else 100))
    imgsz = int(yolo_cfg.get("imgsz", 512))
    batch = int(yolo_cfg.get("smoke_batch_size" if smoke else "batch_size", 2 if smoke else 8))
    workers = int(yolo_cfg.get("workers", 0 if smoke else 4))
    device = cfg.get("runtime", {}).get("device", "auto")
    if device == "auto":
        device = "0"
    model = yolo_cfg.get("base_model", "yolov8s-seg.pt")
    run_name = "smoke_yolo_nuclei" if smoke else "yolo_nuclei"
    project = dirs["run"] / "yolo_runs"

    cmd = [
        sys.executable,
        "-m",
        "ultralytics",
        "segment",
        "train",
        f"model={model}",
        f"data={data_yaml}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"workers={workers}",
        f"device={device}",
        f"project={project}",
        f"name={run_name}",
        "exist_ok=True",
        "classes=0",
    ]
    if smoke:
        cmd.extend(["patience=1", "save_period=1"])

    log_path = dirs["logs"] / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        raise PipelineError("training", f"YOLO nuclei training failed with return code {proc.returncode}. See {log_path}")

    run_dir = project / run_name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, dirs["checkpoints"] / ("smoke_yolo_nuclei_best.pt" if smoke else "yolo_nuclei_best.pt"))
    summary = {
        "stage": "smoke_train_nuclei" if smoke else "train_nuclei",
        "command": " ".join(str(x) for x in cmd),
        "log": str(log_path),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best) if best.exists() else "",
    }
    write_json(dirs["metrics"] / f"{run_name}_summary.json", summary)
    if (run_dir / "results.csv").exists():
        plot_history(run_dir / "results.csv", dirs["run"] / "plots")
    return summary


def smoke_forward_backward(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        checkpoint = dirs["checkpoints"] / "smoke_tiny_checkpoint_no_torch.json"
        summary = {
            "checkpoint": str(checkpoint),
            "loss": "",
            "device": "no_torch",
            "warning": "Torch is not installed, so smoke skipped forward/backward. SUTD config requires CUDA and will fail before this point if CUDA/PyTorch is unavailable.",
        }
        checkpoint.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    tracker = ProgressTracker("smoke_forward_backward", total_epochs=1, total_batches=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1)).to(device)
    x = torch.rand(1, 3, 64, 64, device=device)
    y = torch.rand(1, 1, 64, 64, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()
    checkpoint = dirs["checkpoints"] / "smoke_tiny_torch_checkpoint.pt"
    torch.save({"model": model.state_dict(), "loss": float(loss.detach().cpu())}, checkpoint)
    runtime = tracker.write_runtime_summary(dirs["logs"] / "runtime_summary.json", 1, 1)
    print_epoch_progress(
        1,
        1,
        1,
        1,
        runtime,
        {
            "learning_rate": 1e-3,
            "train_loss": round(float(loss.detach().cpu()), 6),
            "val_loss": "",
            "val_dice": "",
            "val_iou": "",
            "best_val_dice": "",
        },
        checkpoint,
    )
    return {"checkpoint": str(checkpoint), "loss": float(loss.detach().cpu()), "device": str(device)}


def run(cfg: dict[str, Any], dirs: dict[str, Path]) -> dict[str, Any]:
    return train(cfg, dirs, smoke=False)
