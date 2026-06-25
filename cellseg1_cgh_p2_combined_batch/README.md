# Combined CGH P2 CellSeg1 Training Batch

This directory combines the original CGH P2 training batch and the second
PNG-derived compact-cell supplemental batch into one training entrypoint.

The image, mask, auxiliary, semantic, and preview PNG files are relative
symlinks back to the existing batch folders in this repository. This avoids
duplicating large binary assets while keeping one combined manifest.

## Summary

- Tiles: 41
- Trainable cell instances: 1050
- Clear cells: 673 (64.095%)
- Compact cells: 377 (35.905%)
- Stroma regions: 168
- Cell instance CSV rows: 1050
- Boundary QC CSV rows: 1124

## Main Files

- `dataset_manifest.csv`
- `cell_instances.csv`
- `boundary_qc.csv`
- `batch_membership.csv`
- `train/images/`
- `train/masks/`
- `auxiliary_masks/`
- `semantic_masks/`
- `previews/`

Use this directory as the dataset root on the GPU cluster after cloning the repo.
