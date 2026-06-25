"""Run-folder creation and reproducibility snapshots."""

from __future__ import annotations

import importlib
import json
import os
import platform
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.utils.io import PipelineError, repo_root, resolve_path


RUN_SUBDIRS = ("configs", "logs", "checkpoints", "predictions", "overlays", "csv", "metrics", "plots", "tensorboard")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def timestamped_run_dir(cfg: dict[str, Any]) -> Path:
    output_root = resolve_path(cfg, cfg["paths"]["output_root"])
    experiment_name = cfg.get("experiment", {}).get("name", "adrenal_morphology")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in experiment_name)
    run_dir = output_root / f"{stamp}_{safe_name}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"{stamp}_{safe_name}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def run_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {"run": run_dir}
    for subdir in RUN_SUBDIRS:
        dirs[subdir] = run_dir / subdir
    return dirs


def copy_config_files(cfg: dict[str, Any], dirs: dict[str, Path]) -> None:
    config_dir = dirs["configs"]
    merged = {key: value for key, value in cfg.items() if not key.startswith("_")}
    (config_dir / "merged_config.yaml").write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    for key, name in (("_config_path", "input_config.yaml"), ("_thresholds_config_path", "qc_thresholds.yaml")):
        src = cfg.get(key)
        if src and Path(src).exists():
            shutil.copy2(src, config_dir / name)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    packages = packages or ["numpy", "pandas", "PIL", "yaml", "torch", "ultralytics", "cv2"]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            versions[name] = f"missing: {type(exc).__name__}"
    return versions


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", "")
        info["available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count()) if info["available"] else 0
        if info["available"]:
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        info["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as exc:
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


def save_reproducibility_snapshot(cfg: dict[str, Any], dirs: dict[str, Path]) -> None:
    snapshot = {
        "git_commit": git_commit_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "seed": cfg.get("experiment", {}).get("seed"),
        "package_versions": package_versions(),
        "cuda": cuda_info(),
        "environment": {
            key: os.environ.get(key, "")
            for key in ("CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "WANDB_MODE", "HOSTNAME")
        },
    }
    (dirs["configs"] / "reproducibility.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def require_cuda_if_configured(cfg: dict[str, Any]) -> None:
    if not cfg.get("runtime", {}).get("require_cuda", False):
        return
    info = cuda_info()
    if not info.get("available"):
        raise PipelineError("cuda", f"CUDA is required by config but not available. CUDA info: {info}")
