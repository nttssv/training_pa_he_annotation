# CGH P2 CellSeg1 Second Batch

This folder contains PNG-derived training datapoints reconstructed from human
before/after annotation screenshots.

Current datapoints: 6

- `human_compact_tile_001`: 131 trainable cell instances (3 clear, 128 compact), 7 stroma regions.
- `human_compact_tile_002`: 58 trainable cell instances (0 clear, 58 compact), 2 stroma regions.
- `human_compact_tile_003`: 17 trainable cell instances (0 clear, 17 compact), 3 stroma regions.
- `human_compact_tile_004`: 6 trainable cell instances (0 clear, 6 compact), 5 stroma regions.
- `human_compact_tile_005`: 35 trainable cell instances (0 clear, 35 compact), 5 stroma regions.
- `human_compact_tile_006`: 59 trainable cell instances (48 clear, 11 compact), 7 stroma regions.

Important caveats:

- This is not a canonical QuPath vector export.
- The instance mask is reconstructed from colored overlay lines, so review
  `previews/human_compact_tile_006_overlay.png` before training.
- Nucleus objects are not available from the screenshot; nucleus QC fields are
  marked `PNG_DERIVED_NO_NUCLEUS_QC`.
- Use this batch as supplemental data or convert future human annotation pairs
  with `tools/create_png_derived_datapoint.py`.
