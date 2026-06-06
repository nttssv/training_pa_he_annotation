# Nuisal/cellseg1 Analysis For CGH PA Ground Truth

Repository analyzed locally at:

`/Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing/qupath_work/cellseg1_repo`

Commit analyzed: `1c027c2 add cpu support`.

## Fit for this project

CellSeg1 is a reasonable pilot choice for the P2 QuPath ground truth because it fine-tunes SAM with LoRA from a small annotated image set. Our current P2 export has 34 training tiles and 738 valid cell-boundary instances, each checked to contain one nucleus inside the training tile.

This is enough for pipeline iteration and early model comparison, but it is still not enough for a robust final histology model. For a dependable model, add more training tiles covering staining variation, dense cells, stroma-heavy zones, tile edges, and ambiguous/uncertain cases.

## Expected input format

The repo expects paired folders:

- `train/images/*`: RGB or grayscale microscopy image files.
- `train/masks/*`: instance masks where `0` is background and each cell object has a unique positive integer label.

Our exported dataset matches this format:

- Images: `train/images/*.png`
- Masks: `train/masks/*.png`
- Tiles: 34 exported training tiles
- Labels: contiguous `1..N` per tile
- Sizes: mostly `512 x 512`; `p2_tile_01` is `513 x 512`, `p2_tile_07` is `512 x 518`, and `yolo_tile_22` is `513 x 512`.

## What CellSeg1 will use from our QuPath project

Used directly for training:

- `GT Cell boundary` objects, exported as instance labels.

Not used directly by the current repo:

- `GT Nucleus`, except for our export QC that each cell boundary encloses one nucleus.
- `GT Stroma`, except as visual/QC context.
- `GT Uncertain`, except as an auxiliary ignore mask that the repo does not currently read.

## Main repo limitation for our data

The default CellSeg1 sampler has no ignore-mask support. It samples positive points from labeled instances and negative points from all zero/background pixels. In our dataset, some zero pixels are true background/stroma, but some are intentionally uncertain regions. If `neg_rate > 0`, uncertain cell-like areas could become negative examples.

For the first run, the provided config sets:

`neg_rate: 0.0`

This avoids teaching the model that uncertain regions are background. A better second step is to patch the repo so it loads `GT Uncertain` as an ignore mask and samples negative points only outside ignored regions.

## Training entry point

`cellseg1_train.py` exposes a Python function:

`main(config_path)`

It does not provide a direct CLI wrapper. I added a local launcher:

`train_cellseg1_cgh_p2.py`

## Hardware/environment notes

The README recommends NVIDIA GPU with at least 8GB VRAM. The repo has CPU fallback, but SAM ViT-H training on CPU will be very slow. On this machine's default Python, the required packages are not currently installed (`torch`, `opencv-python`, `albumentations`, `pyyaml`, etc.).

For practical training, use a CUDA environment if available. If training on Apple Silicon, the repo would need extra work to use `mps`; the current device selection only checks CUDA and otherwise falls back to CPU.

## Files prepared

- `cgh_p2_cellseg1_config.yaml`: first-run config for the P2 export.
- `train_cellseg1_cgh_p2.py`: terminal launcher calling `cellseg1_train.main(config)`.
- `preview_instance_overlay.png`: visual QC of exported instance labels.
- `TRAINING_NOTES.md`: setup and run instructions.
