"""Run CellSeg1 LoRA training from a YAML config.

This wrapper keeps the cluster notebook simple and avoids relying on the
CellSeg1 repository being installed as a package.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

import yaml


def log(message: str, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if fields:
        suffix = " " + json.dumps(fields, default=str, sort_keys=True)
    print(f"[{timestamp}] {message}{suffix}", flush=True)


def path_info(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    p = Path(path).expanduser()
    info: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if p.exists():
        stat = p.stat()
        info.update(
            {
                "size_bytes": stat.st_size,
                "size_gb": round(stat.st_size / (1024**3), 3),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return info


def count_pngs(path: str | Path | None) -> int | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return len([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".png" and not x.name.startswith("._")])


def read_proc_status(pid: int) -> dict[str, Any]:
    proc_root = Path(f"/proc/{pid}")
    status: dict[str, Any] = {"pid": pid, "exists": proc_root.exists()}
    if not proc_root.exists():
        return status
    for key in ["wchan", "cmdline"]:
        try:
            if key == "cmdline":
                status[key] = (proc_root / key).read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            else:
                status[key] = (proc_root / key).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            pass
    try:
        for line in (proc_root / "status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("State:", "Threads:", "VmRSS:", "voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:")):
                key, value = line.split(":", 1)
                status[key] = value.strip()
    except OSError:
        pass
    try:
        io_lines = (proc_root / "io").read_text(encoding="utf-8", errors="replace").splitlines()
        for line in io_lines:
            if line.startswith(("read_bytes:", "write_bytes:", "syscr:", "syscw:")):
                key, value = line.split(":", 1)
                status[key] = int(value.strip())
    except OSError:
        pass
    return status


def nvidia_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # pragma: no cover - cluster diagnostic only
        return f"nvidia-smi unavailable: {exc}"
    return result.stdout.strip() or result.stderr.strip() or "nvidia-smi returned no output"


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def current_lr(optimizer: Any) -> float | None:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return None


def start_heartbeat(stop_event: Event, config: dict[str, Any], started: float) -> Thread:
    interval = int(os.getenv("CELLSEG1_HEARTBEAT_SECONDS", "30"))
    result_path = Path(config["result_pth_path"])

    def run() -> None:
        while not stop_event.wait(interval):
            log(
                "HEARTBEAT",
                elapsed_min=round((time.time() - started) / 60, 2),
                process=read_proc_status(os.getpid()),
                gpu=nvidia_snapshot(),
                checkpoint=path_info(result_path),
            )

    thread = Thread(target=run, name="cellseg1-debug-heartbeat", daemon=True)
    thread.start()
    return thread


def wrap_step(module: Any, name: str, after: Callable[[Any], dict[str, Any]] | None = None) -> None:
    original = getattr(module, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.time()
        log(f"START {name}")
        try:
            result = original(*args, **kwargs)
        except Exception:
            log(f"FAILED {name}", elapsed_sec=round(time.time() - started, 3))
            traceback.print_exc()
            raise
        fields = {"elapsed_sec": round(time.time() - started, 3)}
        if after is not None:
            try:
                fields.update(after(result))
            except Exception as exc:
                fields["after_hook_error"] = repr(exc)
        log(f"DONE {name}", **fields)
        return result

    setattr(module, name, wrapped)


def install_cellseg1_debug_hooks(cellseg1_train: Any) -> None:
    wrap_step(cellseg1_train, "set_env")
    wrap_step(cellseg1_train, "prepare_directories")
    wrap_step(cellseg1_train, "load_dataset", after=lambda dataset: {"dataset_len": len(dataset)})
    wrap_step(cellseg1_train, "load_model", after=lambda model: {"model_class": model.__class__.__name__})

    def after_setup(result: Any) -> dict[str, Any]:
        trainloader, optimizer, scheduler = result
        return {
            "trainloader_len": len(trainloader),
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
        }

    wrap_step(cellseg1_train, "setup_training", after=after_setup)
    wrap_step(cellseg1_train, "save_model_pth")

    epoch_state = {"epoch": 0}
    loss_log_interval = int(os.getenv("CELLSEG1_LOSS_LOG_INTERVAL", "10"))

    def train_epoch_debug(
        model: Any,
        config: dict[str, Any],
        trainloader: Any,
        optimizer: Any,
        scheduler: Any,
        stop_event: Any = None,
    ) -> None:
        epoch_state["epoch"] += 1
        epoch = epoch_state["epoch"]
        started = time.time()
        log("START train_epoch", epoch=epoch)
        model.train()
        actual_ga_step = 0
        losses: list[float] = []
        skipped_batches = 0
        total_batches = len(trainloader)
        history_path = Path(config["result_pth_path"]).parent / "cellseg1_training_history.csv"

        try:
            for i_batch, batch_data in enumerate(cellseg1_train.tqdm(trainloader, desc="Batches", leave=False)):
                if stop_event is not None and stop_event.is_set():
                    log("STOP train_epoch", epoch=epoch, reason="stop_event")
                    return

                images, true_instance_masks, cell_masks, all_points, all_cell_probs = batch_data

                if not cellseg1_train.is_valid_batch(images, all_points):
                    skipped_batches += 1
                    continue

                batch_images, batch_points = cellseg1_train.to_tensor(
                    images,
                    all_points,
                    config["sam_image_size"],
                )
                loss = cellseg1_train.compute_loss(
                    model,
                    config,
                    batch_images,
                    batch_points,
                    cell_masks,
                    all_points,
                    all_cell_probs,
                )
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)

                actual_ga_step += 1
                denominator = actual_ga_step if (i_batch + 1) == total_batches else config["gradient_accumulation_step"]
                loss_ga = loss / denominator
                loss_ga.backward()

                did_step = False
                if ((i_batch + 1) % config["gradient_accumulation_step"] == 0) or ((i_batch + 1) == total_batches):
                    optimizer.step()
                    optimizer.zero_grad()
                    actual_ga_step = 0
                    scheduler.step()
                    did_step = True

                batch_num = i_batch + 1
                should_log = (
                    batch_num == 1
                    or batch_num == total_batches
                    or (loss_log_interval > 0 and batch_num % loss_log_interval == 0)
                    or did_step
                )
                if should_log:
                    log(
                        "BATCH train_loss",
                        epoch=epoch,
                        batch=batch_num,
                        total_batches=total_batches,
                        loss=round(loss_value, 6),
                        epoch_avg_loss=round(sum(losses) / len(losses), 6),
                        lr=current_lr(optimizer),
                        optimizer_step=did_step,
                        skipped_batches=skipped_batches,
                    )
        except Exception:
            log("FAILED train_epoch", epoch=epoch, elapsed_sec=round(time.time() - started, 3))
            traceback.print_exc()
            raise

        elapsed = time.time() - started
        epoch_summary = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 3),
            "total_batches": total_batches,
            "valid_batches": len(losses),
            "skipped_batches": skipped_batches,
            "train_loss": round(sum(losses) / len(losses), 8) if losses else None,
            "train_loss_min": round(min(losses), 8) if losses else None,
            "train_loss_max": round(max(losses), 8) if losses else None,
            "train_loss_last": round(losses[-1], 8) if losses else None,
            "lr": current_lr(optimizer),
        }
        append_csv_row(history_path, epoch_summary)
        log("DONE train_epoch", **epoch_summary, history_csv=str(history_path))

    cellseg1_train.train_epoch = train_epoch_debug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to a cloned Nuisal/cellseg1 repo.")
    parser.add_argument("--config", required=True, type=Path, help="Runtime CellSeg1 YAML config.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"CellSeg1 repo not found: {repo}")
    if not config_path.exists():
        raise FileNotFoundError(f"CellSeg1 config not found: {config_path}")

    faulthandler.enable(file=sys.stdout, all_threads=True)
    traceback_interval = int(os.getenv("CELLSEG1_TRACEBACK_INTERVAL_SECONDS", "120"))
    if traceback_interval > 0:
        faulthandler.dump_traceback_later(traceback_interval, repeat=True, file=sys.stdout)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    started = time.time()
    stop_event = Event()
    heartbeat = start_heartbeat(stop_event, config, started)

    log("CellSeg1 wrapper config")
    print(
        json.dumps(
            {
                "repo": str(repo),
                "config": str(config_path),
                "train_image_dir": config.get("train_image_dir"),
                "train_mask_dir": config.get("train_mask_dir"),
                "train_image_count": count_pngs(config.get("train_image_dir")),
                "train_mask_count": count_pngs(config.get("train_mask_dir")),
                "result_pth_path": config.get("result_pth_path"),
                "result_pth_info": path_info(config.get("result_pth_path")),
                "model_path": config.get("model_path"),
                "model_path_info": path_info(config.get("model_path")),
                "epoch_max": config.get("epoch_max"),
                "batch_size": config.get("batch_size"),
                "gradient_accumulation_step": config.get("gradient_accumulation_step"),
                "num_workers": config.get("num_workers"),
                "duplicate_data": config.get("duplicate_data"),
                "train_id": config.get("train_id"),
                "pid": os.getpid(),
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )

    sys.path.insert(0, str(repo))
    log("START import cellseg1_train")
    import cellseg1_train

    log("DONE import cellseg1_train")
    install_cellseg1_debug_hooks(cellseg1_train)

    try:
        log("START CellSeg1 main")
        cellseg1_train.main(config_path)
        log("DONE CellSeg1 main")
    finally:
        stop_event.set()
        heartbeat.join(timeout=2)
        if traceback_interval > 0:
            faulthandler.cancel_dump_traceback_later()

    elapsed = time.time() - started
    result_path = Path(config["result_pth_path"])
    summary = {
        "repo": str(repo),
        "config": str(config_path),
        "result_pth_path": str(result_path),
        "result_exists": result_path.exists(),
        "result_info": path_info(result_path),
        "elapsed_seconds": elapsed,
    }
    log("CellSeg1 training summary")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
