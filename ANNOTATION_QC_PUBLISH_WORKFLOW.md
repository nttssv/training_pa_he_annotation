# Annotation QC And Data Publish Workflow

Use this workflow after each QuPath annotation round. The intended sequence is:

1. Create one new 512x512 training tile in QuPath when needed.
2. Annotate the new tile.
3. QC and clean exactly one newly annotated tile in QuPath.
4. Save the QuPath project.
5. Re-export the training data folder.
6. Regenerate the YOLO segmentation dataset.
7. Validate the exported data.
8. Commit and push the training data repo.

## Create A New Tile

When QuPath is open, pan to the tissue region you want to annotate and run:

```text
/Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing/qupath_work/create_next_512_training_tile_from_viewer.groovy
```

The script creates the next root-level `GT Training tile` using the current hierarchy. For the current snapshot, the next tile after `yolo_tile_33` should be `yolo_tile_34` with object prefix `yolo_T34`.

Placement:

- If a non-tile annotation is selected, the new 512x512 tile is centered on that object.
- Otherwise, the new 512x512 tile is centered on the current viewer position.
- If the proposed tile overlaps an existing root training tile, the script refuses to create it.

Expected created hierarchy:

```text
yolo_tile_34
├── nuclei_clean
└── cell_boundary_clean
```

## User Prompt Template

Send this prompt after finishing a tile:

```text
I have finished annotating [TILE_NAME] in my QuPath project.

Please QC [TILE_NAME] only, then publish the updated training data folder and prepare/push a GitHub commit if validation passes.

Inputs:
- Tile parent name: [TILE_NAME]
- Object prefix: [PREFIX]

Project:
- QuPath project: /Volumes/T9/CGH_PA_annotation_1/project.qpproj
- Image: target.tiff
- QuPath app: /Applications/QuPath-0.7.0-arm64.app
- QuPath CLI: /Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64
- Working folder: /Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing
- Data folder: /Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2

Expected hierarchy:
[TILE_NAME]
├── nuclei_clean
└── cell_boundary_clean

Please proceed carefully and preserve all existing annotations from previous tiles.
```

Examples:

```text
Tile parent name: yolo_tile_34
Object prefix: yolo_T34
```

```text
Tile parent name: P2 tile 21
Object prefix: P2_T21
```

## Preflight

Before modifying QuPath project files:

1. Check whether QuPath is open.
2. Ask the user to save and close QuPath if it is open.
3. Back up:
   - `/Volumes/T9/CGH_PA_annotation_1/project.qpproj`
   - `/Volumes/T9/CGH_PA_annotation_1/data/1/data.qpdata`
   - `/Volumes/T9/CGH_PA_annotation_1/classifiers/classes.json` only if class definitions will be changed.
4. Confirm current Git state in `/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2`.

Do not edit the QuPath project if the backup fails.

## QuPath Editing Rules

Do not use:

```groovy
hierarchy.resolveHierarchy()
```

Use direct parent manipulation only:

```groovy
parent.removeChildObject(obj)
tile.addChildObject(obj)
group.addChildObject(obj)
getCurrentHierarchy().fireHierarchyChangedEvent(this, tile)
```

Scope:

- QC and clean `[TILE_NAME]` only.
- Do not move objects from previous tiles unless they clearly belong to `[TILE_NAME]`.
- Find root-level annotations, temporary `yolo_tile` / `yolo_tile_vXX` groups, or root-level `nuclei_clean` / `cell_boundary_clean` groups that clearly belong to `[TILE_NAME]`.
- Merge those objects into the correct `[TILE_NAME]` hierarchy.
- Keep edge objects under `[TILE_NAME]` if they were clearly annotated for this tile, and flag them as edge/outside-tile.
- Do not clip/cut manually drawn cell boundaries to the tile rectangle unless explicitly requested.
- Do not over-smooth manually annotated boundaries.

## Classes And Naming

Classes:

- Nuclei: `GT Nucleus`
- Clear cell boundary: `GT Clear cell boundary`
- Compact cell boundary: `GT Compact cell boundary`
- Uncertain clear cell boundary: `GT Uncertain clear cell boundary`
- Stroma: `GT Stroma`
- Tile parent and group parents: `GT Training tile`

Rename objects:

- `[PREFIX]_Nucleus_001`, `[PREFIX]_Nucleus_002`, ...
- `[PREFIX]_ClearCellBoundary_001`, ...
- `[PREFIX]_CompactCellBoundary_001`, ...
- `[PREFIX]_UncertainCellBoundary_001`, ...
- `[PREFIX]_Stroma_01`, `[PREFIX]_Stroma_02`, ...

## QC Rules

1. Each `GT Clear cell boundary` or `GT Compact cell boundary` should contain exactly one `GT Nucleus` centroid.
2. If a clear/compact boundary contains zero nuclei, keep it but reclassify it as `GT Uncertain clear cell boundary`.
3. If a boundary contains more than one nucleus, keep it and flag `review_required` / `multiple_nuclei_inside`.
4. If a nucleus is not inside any cell boundary, keep it and flag `orphan_nucleus`.
5. If a nucleus centroid is inside `GT Stroma`, flag `ignore_for_export` / `nucleus_inside_stroma`.
6. Cell boundaries should not overlap each other. Trim overlap carefully without over-smoothing and without clipping to the tile.
7. Stroma should not overlap cell boundaries. Trim stroma against cell boundaries if needed.
8. Nucleus can exist inside a cell boundary; do not subtract nucleus from cell boundary.
9. Edge cells can be kept, but flag `edge/outside-tile` if the object or nucleus is outside the tile.
10. Do not over-smooth manually annotated boundaries.

## Required QC Report

After QC, report:

- Number of nuclei
- Number of clear cell boundaries
- Number of compact cell boundaries
- Number of uncertain clear cell boundaries
- Number of stroma regions
- Boundary overlap pairs
- Stroma-boundary overlap pairs
- Boundaries with zero nuclei
- Boundaries with more than one nucleus
- Orphan nuclei
- Nuclei inside stroma
- Edge/outside-tile objects
- Whether `[TILE_NAME]` is ready for training export

Do not publish if `[TILE_NAME]` is not ready for training export.

## QuPath CLI

Use QuPath CLI when running QC/export scripts:

```bash
"/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64" script -s \
  -p "/Volumes/T9/CGH_PA_annotation_1/project.qpproj" \
  -i "target.tiff" \
  SCRIPT.groovy
```

After QC succeeds:

1. Save the QuPath project.
2. Reopen QuPath for user visual review if needed.
3. Run the canonical export script.

Canonical export:

```bash
"/Applications/QuPath-0.7.0-arm64.app/Contents/MacOS/QuPath-0.7.0-arm64" script -s \
  -p "/Volumes/T9/CGH_PA_annotation_1/project.qpproj" \
  -i "target.tiff" \
  "/Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing/qupath_work/cellseg1_export_cgh_p2.groovy"
```

The export regenerates the full CellSeg1 training data snapshot, not just one tile.

After QuPath export succeeds, regenerate the YOLO segmentation dataset:

```bash
/opt/anaconda3/bin/python \
  "/Users/nttssv/Desktop/SUTD/Courses/CGH PA/CGH Digital Image/Cellpose_testing/training/convert_cellseg1_cgh_p2_to_yolo.py" \
  --source "/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2" \
  --output "/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2/yolo_seg_dataset"
```

## Data Publish Validation

Before commit/push, validate:

- `dataset_manifest.csv` row count equals the current expected tile count.
- Every manifest image path exists.
- Every manifest mask path exists.
- Image and mask sizes match the manifest.
- Mask labels are contiguous from `1..N`, where `N=trainable_instances`.
- Per-tile `*_instances.csv` row counts match `trainable_instances`.
- `cell_instances.csv` rows match total trainable instances.
- `boundary_qc.csv` included boundaries match total trainable instances.
- `yolo_seg_dataset/data.yaml` exists.
- YOLO label files exist for every tile in `yolo_seg_dataset/images`.

Run:

```bash
cd "/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2"
python3 validate_training_export.py --expected-tiles [EXPECTED_TILE_COUNT]
```

Current snapshot: `33` tiles. If `[TILE_NAME]` is a new final training tile, increment the expected count after export.

Clean or ignore before commit:

- AppleDouble files: `._*`
- `.DS_Store`
- QuPath/OS temp files
- training runs: `yolo_runs/`
- derived oversampled datasets: `yolo_seg_dataset_cv_oversampled/`
- model weights and exports: `*.pt`, `*.pth`, `*.onnx`, `*.engine`

## GitHub Publish

Only commit the training-data snapshot after QC and export validation pass.

Recommended commit message format:

```text
Update training data after QC [TILE_NAME]
```

Recommended commands:

```bash
cd "/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2"
git status --short
git add .
git status --short
git commit -m "Update training data after QC [TILE_NAME]"
git push
```

If the folder is not a Git repo yet, initialize/connect it once before the first publish:

```bash
cd "/Volumes/T9/CGH_PA_annotation_1/training_data/cellseg1_cgh_p2"
git init
git remote add origin [GITHUB_REPO_URL]
git branch -M main
git add .
git commit -m "Initial CGH P2 training data snapshot"
git push -u origin main
```

Do not run `git push` if there is no configured remote or if validation fails.
