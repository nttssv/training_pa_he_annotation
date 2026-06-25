"""Visualization helpers for dataset previews, overlays, plots, and HTML reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.utils.geometry import mask_boundary, read_mask


COLORS = {
    "nucleus": np.array([30, 90, 255], dtype=np.uint8),
    "clear": np.array([30, 190, 90], dtype=np.uint8),
    "compact": np.array([245, 210, 45], dtype=np.uint8),
    "uncertain": np.array([255, 145, 30], dtype=np.uint8),
    "edge": np.array([255, 105, 40], dtype=np.uint8),
    "false_positive": np.array([230, 40, 40], dtype=np.uint8),
    "false_negative": np.array([150, 60, 210], dtype=np.uint8),
    "boundary": np.array([30, 190, 90], dtype=np.uint8),
}


def load_rgb(path: Path, max_size: int | None = None) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_size is not None:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return np.asarray(img)


def resize_mask_like(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint16)
    if mask.shape[:2] == image.shape[:2]:
        return mask
    pil = Image.fromarray(mask.astype(np.uint16))
    return np.asarray(pil.resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))


def overlay_masks(
    image_path: Path,
    out_path: Path,
    boundary_mask: Path | None = None,
    nucleus_mask: Path | None = None,
    clear_mask: Path | None = None,
    compact_mask: Path | None = None,
    uncertain_mask: Path | None = None,
    edge_mask: Path | None = None,
    title: str | None = None,
    max_size: int | None = 768,
) -> Path:
    img = load_rgb(image_path, max_size=max_size)
    out = img.copy()

    def draw(mask_path: Path | None, color_name: str, alpha: float = 0.25, boundary_only: bool = True) -> None:
        nonlocal out
        mask = resize_mask_like(read_mask(mask_path, img.shape[:2]), img)
        if mask.size == 0 or not np.any(mask):
            return
        binary = mask > 0
        if not boundary_only:
            overlay = out.copy()
            overlay[binary] = COLORS[color_name]
            out = ((1 - alpha) * out + alpha * overlay).astype(np.uint8)
        boundary = mask_boundary(binary)
        out[boundary] = COLORS[color_name]
        # Thicken boundaries by one pixel for visibility.
        ys, xs = np.where(boundary)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            yy = np.clip(ys + dy, 0, out.shape[0] - 1)
            xx = np.clip(xs + dx, 0, out.shape[1] - 1)
            out[yy, xx] = COLORS[color_name]

    draw(boundary_mask, "boundary", alpha=0.16, boundary_only=True)
    draw(clear_mask, "clear", alpha=0.16, boundary_only=False)
    draw(compact_mask, "compact", alpha=0.16, boundary_only=False)
    draw(nucleus_mask, "nucleus", alpha=0.20, boundary_only=False)
    draw(uncertain_mask, "uncertain", alpha=0.16, boundary_only=False)
    draw(edge_mask, "edge", alpha=0.16, boundary_only=False)

    pil = Image.fromarray(out)
    if title:
        pil = add_title(pil, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    return out_path


def add_title(img: Image.Image, title: str) -> Image.Image:
    pad = 28
    canvas = Image.new("RGB", (img.width, img.height + pad), "white")
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(20, 20, 20))
    return canvas


def make_image_grid(items: list[tuple[Path, str]], out_path: Path, columns: int = 4, thumb_size: int = 220) -> Path:
    if not items:
        raise ValueError("No images provided for grid")
    thumbs: list[Image.Image] = []
    for path, label in items:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_size, thumb_size + 24), "white")
            tile.paste(img, ((thumb_size - img.width) // 2, 0))
            draw = ImageDraw.Draw(tile)
            draw.text((4, thumb_size + 5), label[:36], fill=(20, 20, 20))
            thumbs.append(tile)
    rows = int(np.ceil(len(thumbs) / columns))
    grid = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + 24)), "white")
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, columns)
        grid.paste(thumb, (col * thumb_size, row * (thumb_size + 24)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path


def bar_chart(counts: dict[str, int], out_path: Path, title: str) -> Path:
    import matplotlib.pyplot as plt

    labels = list(counts) or ["none"]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    ax.bar(labels, values, color="#4a78a8")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_history(history_csv: Path, plots_dir: Path) -> list[Path]:
    import pandas as pd
    import matplotlib.pyplot as plt

    if not history_csv.exists():
        return []
    df = pd.read_csv(history_csv)
    outputs: list[Path] = []
    series = {
        "train_loss_curve.png": "train_loss",
        "val_loss_curve.png": "val_loss",
        "val_dice_curve.png": "val_dice",
        "val_iou_curve.png": "val_iou",
        "learning_rate_curve.png": "learning_rate",
        "valid_cell_rate_curve.png": "valid_cell_rate",
        "uncertain_cell_rate_curve.png": "uncertain_cell_rate",
    }
    for filename, column in series.items():
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["epoch"], df[column], marker="o")
        ax.set_xlabel("epoch")
        ax.set_ylabel(column)
        ax.set_title(column.replace("_", " "))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path = plots_dir / filename
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def write_html_report(
    out_path: Path,
    title: str,
    sections: dict[str, Any],
) -> Path:
    def img_tag(path: str | Path | None) -> str:
        if not path:
            return ""
        p = Path(path)
        rel = p.name if p.parent == out_path.parent else str(p)
        return f'<figure><img src="{html.escape(rel)}" style="max-width:100%;"><figcaption>{html.escape(p.name)}</figcaption></figure>'

    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.35}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:4px 8px}"
        "img{border:1px solid #ddd;margin:4px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}"
        "pre{background:#f6f6f6;padding:12px;overflow:auto}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for heading, content in sections.items():
        body.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, dict):
            body.append("<table>")
            for key, value in content.items():
                body.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
            body.append("</table>")
        elif isinstance(content, list):
            body.append("<div class='grid'>")
            for item in content:
                body.append(img_tag(item))
            body.append("</div>")
        else:
            body.append(f"<pre>{html.escape(str(content))}</pre>")
    body.append("</body></html>")
    out_path.write_text("\n".join(body), encoding="utf-8")
    return out_path

