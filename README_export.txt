CGH P2 ground-truth export for CellSeg1 and downstream morphometry models

Source QuPath project: /Volumes/T9/CGH_PA_annotation_1/project.qpproj
Source image: target.tiff
Export date: 2026-06-06
Training tiles: final GT Training tile annotations named P2 tile NN or yolo_tile_NN.
Temporary/duplicate training tile annotations such as cell_boundary_clean, nuclei_clean, and unnumbered yolo_tile are intentionally excluded.

Primary CellSeg1 files:
- train/images/*.png: one image per exported training tile
- train/masks/*.png: uint8 instance masks, 0=background, 1..N=trainable cell boundary instances

Auxiliary masks:
- auxiliary_masks/*_gt_nucleus_instances.png: in-tile GT Nucleus instance masks, excluding nuclei marked ignore or located inside GT Stroma
- auxiliary_masks/*_gt_nucleus_all_direct_children.png: direct child nuclei, including edge/outside-tile nuclei kept for review, excluding nuclei marked ignore or located inside GT Stroma
- auxiliary_masks/*_gt_stroma.png: GT Stroma binary mask, including stroma crossing the tile edge
- auxiliary_masks/*_gt_uncertain_ignore.png: GT Uncertain clear cell boundary regions for ignore/review handling
- auxiliary_masks/*_edge_or_invalid_cell_ignore.png: clear/compact cell boundaries excluded from CellSeg1 positives because the nucleus is outside the tile or the one-nucleus rule fails
- auxiliary_masks/*_gt_clear_boundary_all.png: all clear-cell boundary annotations for the tile
- auxiliary_masks/*_gt_compact_boundary_all.png: all compact-cell boundary annotations for the tile
- semantic_masks/*_semantic_review.png: review-only semantic mask; 1=clear boundary, 2=compact boundary, 3=uncertain, 4=stroma

CSV metadata:
- dataset_manifest.csv: one row per tile
- cell_instances.csv: combined label-to-annotation mapping
- *_instances.csv: per-tile instance mappings
- boundary_qc.csv: boundary inclusion flags and stroma-overlap measurements

Export rules:
- Positive CellSeg1 instances include GT Clear cell boundary and GT Compact cell boundary only.
- A positive boundary must contain exactly one GT Nucleus centroid whose centroid is inside the same training tile.
- A positive boundary is excluded if its nucleus is inside GT Stroma or if exclude_from_training_export=true.
- GT Uncertain clear cell boundary annotations are exported as ignore/review masks, not positive training instances.
- Edge cells with visible boundary but nucleus outside the tile are exported as ignore masks, not positive training instances.
- Stroma masks include GT Stroma that intersects the tile even if the stroma annotation is parented under a neighboring training tile.
- Exported stroma/uncertain/edge-ignore masks are clipped away from positive trainable instances to avoid auxiliary-mask overlap.

Current summary:
- p2_tile_01: 19 trainable instances (19 clear, 0 compact), 0 edge/invalid ignored, 6 uncertain ignored, 19 in-tile nuclei, 6 intersecting stroma regions, size 513x512
- p2_tile_02: 18 trainable instances (16 clear, 2 compact), 0 edge/invalid ignored, 11 uncertain ignored, 18 in-tile nuclei, 5 intersecting stroma regions, size 512x512
- p2_tile_03: 25 trainable instances (25 clear, 0 compact), 1 edge/invalid ignored, 11 uncertain ignored, 25 in-tile nuclei, 5 intersecting stroma regions, size 512x512
- p2_tile_04: 13 trainable instances (13 clear, 0 compact), 3 edge/invalid ignored, 10 uncertain ignored, 22 in-tile nuclei, 6 intersecting stroma regions, size 512x512
- p2_tile_05: 19 trainable instances (19 clear, 0 compact), 3 edge/invalid ignored, 10 uncertain ignored, 22 in-tile nuclei, 7 intersecting stroma regions, size 512x512
- p2_tile_06: 16 trainable instances (16 clear, 0 compact), 2 edge/invalid ignored, 16 uncertain ignored, 17 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_07: 21 trainable instances (19 clear, 2 compact), 1 edge/invalid ignored, 15 uncertain ignored, 21 in-tile nuclei, 5 intersecting stroma regions, size 512x518
- p2_tile_08: 14 trainable instances (14 clear, 0 compact), 4 edge/invalid ignored, 15 uncertain ignored, 17 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_09: 16 trainable instances (16 clear, 0 compact), 2 edge/invalid ignored, 19 uncertain ignored, 24 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_10: 20 trainable instances (20 clear, 0 compact), 2 edge/invalid ignored, 17 uncertain ignored, 20 in-tile nuclei, 6 intersecting stroma regions, size 512x512
- p2_tile_11: 21 trainable instances (19 clear, 2 compact), 0 edge/invalid ignored, 25 uncertain ignored, 36 in-tile nuclei, 6 intersecting stroma regions, size 512x512
- p2_tile_12: 48 trainable instances (24 clear, 24 compact), 2 edge/invalid ignored, 11 uncertain ignored, 54 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- p2_tile_13: 33 trainable instances (29 clear, 4 compact), 7 edge/invalid ignored, 9 uncertain ignored, 35 in-tile nuclei, 7 intersecting stroma regions, size 512x512
- p2_tile_14: 35 trainable instances (26 clear, 9 compact), 11 edge/invalid ignored, 14 uncertain ignored, 37 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_15: 23 trainable instances (17 clear, 6 compact), 2 edge/invalid ignored, 8 uncertain ignored, 24 in-tile nuclei, 6 intersecting stroma regions, size 512x512
- p2_tile_16: 36 trainable instances (19 clear, 17 compact), 6 edge/invalid ignored, 4 uncertain ignored, 37 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_17: 18 trainable instances (16 clear, 2 compact), 7 edge/invalid ignored, 12 uncertain ignored, 20 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- p2_tile_18: 22 trainable instances (14 clear, 8 compact), 3 edge/invalid ignored, 8 uncertain ignored, 24 in-tile nuclei, 6 intersecting stroma regions, size 512x512
- p2_tile_19: 23 trainable instances (22 clear, 1 compact), 0 edge/invalid ignored, 13 uncertain ignored, 23 in-tile nuclei, 8 intersecting stroma regions, size 512x512
- p2_tile_20: 18 trainable instances (17 clear, 1 compact), 3 edge/invalid ignored, 13 uncertain ignored, 20 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_21: 38 trainable instances (19 clear, 19 compact), 2 edge/invalid ignored, 1 uncertain ignored, 38 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_22: 18 trainable instances (9 clear, 9 compact), 5 edge/invalid ignored, 3 uncertain ignored, 22 in-tile nuclei, 1 intersecting stroma regions, size 513x512
- yolo_tile_23: 22 trainable instances (12 clear, 10 compact), 1 edge/invalid ignored, 9 uncertain ignored, 25 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_24: 23 trainable instances (22 clear, 1 compact), 1 edge/invalid ignored, 11 uncertain ignored, 27 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_25: 19 trainable instances (17 clear, 2 compact), 0 edge/invalid ignored, 15 uncertain ignored, 25 in-tile nuclei, 2 intersecting stroma regions, size 512x512
- yolo_tile_26: 16 trainable instances (16 clear, 0 compact), 3 edge/invalid ignored, 9 uncertain ignored, 17 in-tile nuclei, 1 intersecting stroma regions, size 512x512
- yolo_tile_27: 19 trainable instances (19 clear, 0 compact), 0 edge/invalid ignored, 12 uncertain ignored, 22 in-tile nuclei, 2 intersecting stroma regions, size 512x512
- yolo_tile_28: 13 trainable instances (12 clear, 1 compact), 0 edge/invalid ignored, 19 uncertain ignored, 20 in-tile nuclei, 2 intersecting stroma regions, size 512x512
- yolo_tile_29: 15 trainable instances (14 clear, 1 compact), 3 edge/invalid ignored, 11 uncertain ignored, 17 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- yolo_tile_30: 16 trainable instances (16 clear, 0 compact), 0 edge/invalid ignored, 12 uncertain ignored, 17 in-tile nuclei, 4 intersecting stroma regions, size 512x512
- yolo_tile_31: 21 trainable instances (21 clear, 0 compact), 0 edge/invalid ignored, 6 uncertain ignored, 21 in-tile nuclei, 1 intersecting stroma regions, size 512x512
- yolo_tile_32: 23 trainable instances (23 clear, 0 compact), 0 edge/invalid ignored, 16 uncertain ignored, 24 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_33: 20 trainable instances (20 clear, 0 compact), 0 edge/invalid ignored, 21 uncertain ignored, 21 in-tile nuclei, 3 intersecting stroma regions, size 512x512
- yolo_tile_34: 17 trainable instances (17 clear, 0 compact), 0 edge/invalid ignored, 21 uncertain ignored, 17 in-tile nuclei, 2 intersecting stroma regions, size 512x512

Total trainable instances: 738 (617 clear, 121 compact)
Total uncertain ignored regions: 413
Total edge/invalid ignored boundaries: 74
