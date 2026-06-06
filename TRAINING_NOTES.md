# CGH PA P2 CellSeg1 Training Notes

## Current Ground Truth Snapshot

Source project: `/Volumes/T9/CGH_PA_annotation_1/project.qpproj`
Source image: `target.tiff`
Training tiles: 33 final GT tile annotations (`P2 tile 01`-`P2 tile 20` and `yolo_tile_21`-`yolo_tile_33`)
Export path on T9: `/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2`

Current export has 33 tiles and 721 trainable cell-boundary instances:

| Tile | Trainable | Clear | Compact | Edge/invalid ignore | Uncertain ignore | In-tile nuclei | Stroma regions |
|---|---:|---:|---:|---:|---:|---:|---:|
| p2_tile_01 | 19 | 19 | 0 | 0 | 6 | 19 | 6 |
| p2_tile_02 | 18 | 16 | 2 | 0 | 11 | 18 | 5 |
| p2_tile_03 | 25 | 25 | 0 | 1 | 11 | 25 | 5 |
| p2_tile_04 | 13 | 13 | 0 | 3 | 10 | 22 | 6 |
| p2_tile_05 | 19 | 19 | 0 | 3 | 10 | 22 | 7 |
| p2_tile_06 | 16 | 16 | 0 | 2 | 16 | 17 | 4 |
| p2_tile_07 | 21 | 19 | 2 | 1 | 15 | 21 | 5 |
| p2_tile_08 | 14 | 14 | 0 | 4 | 15 | 17 | 4 |
| p2_tile_09 | 16 | 16 | 0 | 2 | 19 | 24 | 4 |
| p2_tile_10 | 20 | 20 | 0 | 2 | 17 | 20 | 6 |
| p2_tile_11 | 21 | 19 | 2 | 0 | 25 | 36 | 6 |
| p2_tile_12 | 48 | 24 | 24 | 2 | 11 | 54 | 3 |
| p2_tile_13 | 33 | 29 | 4 | 7 | 9 | 35 | 7 |
| p2_tile_14 | 35 | 26 | 9 | 11 | 14 | 37 | 4 |
| p2_tile_15 | 23 | 17 | 6 | 2 | 8 | 24 | 6 |
| p2_tile_16 | 36 | 19 | 17 | 6 | 4 | 37 | 4 |
| p2_tile_17 | 18 | 16 | 2 | 7 | 12 | 20 | 4 |
| p2_tile_18 | 22 | 14 | 8 | 3 | 8 | 24 | 6 |
| p2_tile_19 | 23 | 22 | 1 | 0 | 13 | 23 | 8 |
| p2_tile_20 | 18 | 17 | 1 | 3 | 13 | 20 | 3 |
| yolo_tile_21 | 38 | 19 | 19 | 2 | 1 | 38 | 3 |
| yolo_tile_22 | 18 | 9 | 9 | 5 | 3 | 22 | 1 |
| yolo_tile_23 | 22 | 12 | 10 | 1 | 9 | 25 | 3 |
| yolo_tile_24 | 23 | 22 | 1 | 1 | 11 | 27 | 3 |
| yolo_tile_25 | 19 | 17 | 2 | 0 | 13 | 25 | 2 |
| yolo_tile_26 | 16 | 16 | 0 | 3 | 9 | 17 | 1 |
| yolo_tile_27 | 19 | 19 | 0 | 0 | 12 | 22 | 2 |
| yolo_tile_28 | 13 | 12 | 1 | 0 | 19 | 20 | 2 |
| yolo_tile_29 | 15 | 14 | 1 | 3 | 11 | 17 | 4 |
| yolo_tile_30 | 16 | 16 | 0 | 0 | 10 | 17 | 4 |
| yolo_tile_31 | 21 | 21 | 0 | 0 | 5 | 21 | 1 |
| yolo_tile_32 | 23 | 23 | 0 | 0 | 16 | 24 | 3 |
| yolo_tile_33 | 20 | 20 | 0 | 0 | 20 | 21 | 3 |
| Total | 721 | 600 | 121 | 74 | 391 | 811 | - |

Notes:
- Positive CellSeg1 masks use only `GT Clear cell boundary` and `GT Compact cell boundary` regions with exactly one in-tile nucleus centroid.
- `GT Uncertain cell boundary` regions are exported as ignore/review masks, not positive instances.
- Edge cells with nuclei outside the training ROI are exported as ignore masks.
- Temporary/duplicate training tile annotations such as `cell_boundary_clean`, `nuclei_clean`, and unnumbered `yolo_tile` are intentionally excluded from the export.

## Files

Primary CellSeg1 inputs:
- `train/images/*.png`: one image per exported training tile
- `train/masks/*.png`: uint8 instance masks, `0=background`, `1..N=trainable cell boundary instances`

Auxiliary masks:
- `auxiliary_masks/*_gt_nucleus_instances.png`
- `auxiliary_masks/*_gt_nucleus_all_direct_children.png`
- `auxiliary_masks/*_gt_stroma.png`
- `auxiliary_masks/*_gt_uncertain_ignore.png`
- `auxiliary_masks/*_edge_or_invalid_cell_ignore.png`
- `auxiliary_masks/*_gt_clear_boundary_all.png`
- `auxiliary_masks/*_gt_compact_boundary_all.png`
- `semantic_masks/*_semantic_review.png`

Metadata:
- `dataset_manifest.csv`
- `cell_instances.csv`
- `boundary_qc.csv`
- `*_instances.csv`

## Training Direction

Use the 33-tile dataset as the current P2 ground-truth snapshot. It is enough for pipeline iteration and early model comparison, but it is still not enough for a robust final histology model.

Target annotation order for the eventual model:
1. Detect nuclei first.
2. Detect stroma/mesenchyme next.
3. Segment cell boundaries using nuclei as anchors and stroma as exclusion/context.
4. Mark ambiguous cytoplasm as uncertain/ignore rather than forcing a clear/compact label.

Practical route:
1. Fine-tune CellSeg1/SAM LoRA on clear + compact cell-boundary instance masks.
2. Keep `neg_rate: 0.0` until CellSeg1 sampler is patched to honor ignore masks.
3. Train or adapt a separate nucleus detector from `*_gt_nucleus_instances.png`.
4. Train a semantic stroma model from `*_gt_stroma.png`.
5. Combine outputs into the Prototype 1 dashboard metrics.

## Metrics To Track

Segmentation metrics:
- Cell instance AP50/AP75 and mean IoU.
- Boundary Dice/IoU for clear + compact masks.
- Nucleus instance Dice/IoU.
- Stroma semantic Dice/IoU.
- One-nucleus-per-cell rate.
- Cell count absolute error.
- Edge/uncertain false-positive rate.

Dashboard metrics:
- Cells detected.
- Nuclei detected and assigned.
- Parenchyme area and percent.
- Mesenchyme/stroma area and percent.
- Nuclear area, cytoplasm area, and N/C ratio.
- Clear, compact, and uncertain cytoplasm proportions.

## Recommended Dataset Growth

Current snapshot: 33 annotated training tiles.
For a model we can trust across slides: plan for 50+ tiles across variable tissue density, staining intensity, compact cytoplasm, stroma-rich areas, and edge cases.

Daily workflow:
1. Annotate one or two new 512x512 training tiles in QuPath.
2. Ask Codex to QC the new tile using `ANNOTATION_QC_PUBLISH_WORKFLOW.md`.
3. Save the QuPath project after tile-only QC passes.
4. Run the canonical export script to regenerate this training data folder.
5. Regenerate `yolo_seg_dataset` from the CellSeg1 export.
6. Validate the export against `dataset_manifest.csv`, `cell_instances.csv`, `boundary_qc.csv`, and YOLO labels.
7. Commit and push the data repo only after validation passes.

## Commands

Export from QuPath:

```bash
/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64 script -s \
  -p /Volumes/T9/CGH_PA_annotation_1/project.qpproj \
  -i target.tiff \
  /Users/nttssv/Desktop/SUTD/Courses/CGH\ PA/CGH\ Digital\ Image/Cellpose_testing/qupath_work/cellseg1_export_cgh_p2.groovy
```

Regenerate YOLO segmentation data:

```bash
/opt/anaconda3/bin/python /Users/nttssv/Desktop/SUTD/Courses/CGH\ PA/CGH\ Digital\ Image/Cellpose_testing/training/convert_cellseg1_cgh_p2_to_yolo.py \
  --source /Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2 \
  --output /Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/yolo_seg_dataset
```

Train after installing CellSeg1 dependencies and setting `model_path`:

```bash
python3 /Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/train_cellseg1_cgh_p2.py \
  --config /Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/cgh_p2_cellseg1_config.yaml
```
