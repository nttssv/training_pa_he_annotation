# Adrenal H&E Morphology Training Pipeline

This repository now includes a config-driven training and QC pipeline for
measurement-grade adrenal H&E morphology. The target readouts are cell area,
nuclear area, cytoplasm area, N/C ratio, clear-cell fraction, compact-cell
fraction, valid cell rate, uncertain cell rate, and edge rejection rate.

The production design is a two-model workflow:

1. YOLO nuclei segmentation.
2. CellSeg1/SAM-style cell-boundary segmentation.

Nucleus predictions are QC anchors for boundaries. A valid cell must contain
exactly one nucleus. Zero nuclei, multiple nuclei, edge-cut cells,
vessel/stroma overlap, or ambiguous phenotype are exported as uncertain rather
than treated as measurement-grade cells.

## Config-Driven Pipeline

Main entrypoint:

```bash
python training/run_pipeline.py --config training/configs/local.yaml --stage smoke --visualize
```

Supported stages:

```bash
python training/run_pipeline.py --config training/configs/sutd.yaml --stage audit
python training/run_pipeline.py --config training/configs/sutd.yaml --stage split
python training/run_pipeline.py --config training/configs/sutd.yaml --stage visualize
python training/run_pipeline.py --config training/configs/sutd.yaml --stage smoke --visualize
python training/run_pipeline.py --config training/configs/sutd.yaml --stage train_nuclei
python training/run_pipeline.py --config training/configs/sutd.yaml --stage train_boundary --visualize
python training/run_pipeline.py --config training/configs/sutd.yaml --stage infer
python training/run_pipeline.py --config training/configs/sutd.yaml --stage morphology
python training/run_pipeline.py --config training/configs/sutd.yaml --stage all --visualize
```

All paths, model parameters, thresholds, pixel size, batch size, epochs, and
device settings are read from YAML:

- `training/configs/local.yaml`
- `training/configs/sutd.yaml`
- `training/configs/qc_thresholds.yaml`

For SUTD, set paths through environment variables instead of editing code:

```bash
export CGH_DATASET_ROOT=/path/to/cellseg1_cgh_p2_combined_41_full
export CGH_OUTPUT_ROOT=/path/to/outputs/runs
export CELLSEG1_REPO=/path/to/cellseg1
export CELLSEG1_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
```

## Local-To-SUTD Workflow

Clone the lightweight training-code branch on the SUTD GPU server:

```bash
cd ~/Desktop/2.Train2_25_Jun
git clone --branch codex/adrenal-morphology-training-pipeline --single-branch https://github.com/nttssv/training_pa_he_annotation.git
cd training_pa_he_annotation
```

Clone the data branch separately. For CellSeg1/SAM boundary training, use the
full 41-tile dataset folder `cellseg1_cgh_p2_combined_41_full/`:

```bash
mkdir -p ~/Desktop/1.Data
cd ~/Desktop/1.Data
git clone --branch codex/add-second-batch-training-data --single-branch https://github.com/nttssv/training_pa_he_annotation.git training_pa_he_annotation_full
cd training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full
find train/images -type f | wc -l
find train/masks -type f | wc -l
```

Then return to the training-code clone and run:

```bash
cd ~/Desktop/2.Train2_25_Jun/training_pa_he_annotation
export CGH_DATASET_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
export CGH_OUTPUT_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs/runs"

bash training/scripts/run_sutd_smoke_test.sh
bash training/scripts/run_sutd_training.sh
bash training/scripts/run_sutd_inference.sh
```

If YOLO nuclei training is already complete and you only need to continue with
CellSeg1/SAM boundary training, open and run:

```bash
cd ~/Desktop/2.Train2_25_Jun/training_pa_he_annotation
export CGH_DATASET_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full"
export CGH_OUTPUT_ROOT="$HOME/Desktop/1.Data/training_pa_he_annotation_full/outputs"
jupyter lab training/cellseg1_cluster_live_training.ipynb
```

That notebook defaults to:

```text
~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full/train/images
~/Desktop/1.Data/training_pa_he_annotation_full/cellseg1_cgh_p2_combined_41_full/train/masks
```

The full training script runs the smoke test first and stops if smoke fails.
PyTorch should be installed separately with the CUDA build appropriate for the
assigned SUTD node; `requirements.txt` intentionally does not pin torch.
If `CGH_YOLO_DATA_YAML` is not set and the legacy
`training_data/dataset/yolo_seg_dataset/data.yaml` is absent, the YOLO nuclei
stage automatically converts `CGH_DATASET_ROOT` into a run-local YOLO-seg
dataset under the current output run folder.
The SUTD config defaults YOLO dataloader workers to `0` because Jupyter GPU
containers often have small `/dev/shm`; using multiple workers can trigger
PyTorch bus errors even when disk and GPU memory are sufficient.

You can also let the helper script clone/update the data branch:

```bash
bash training/scripts/setup_sutd_data.sh
```

## Smoke Test

```bash
python training/run_pipeline.py --config training/configs/sutd.yaml --stage smoke --visualize
```

Smoke test checks CUDA when required, verifies dataset paths and image-mask
pairs, creates train/val manifests, runs a tiny forward/backward pass when
PyTorch is available, saves a tiny checkpoint, exports overlays, writes
`matched_cells.csv`, and exports morphology CSVs.

## Visualization And Monitoring

Generate dataset preview before training:

```bash
python training/run_pipeline.py --config training/configs/sutd.yaml --stage visualize
```

Outputs include:

- `dataset_preview/train_tiles_grid.png`
- `dataset_preview/val_tiles_grid.png`
- `dataset_preview/class_examples_grid.png`
- `dataset_preview/class_distribution.png`
- `dataset_preview/batch_distribution.png`
- `dataset_preview/annotation_overlay_examples/*_overlay.png`
- `training_report.html`

During smoke/training, progress utilities write:

- `logs/runtime_summary.json`
- `logs/run_summary.txt`
- `logs/error_report.txt` on failure
- `metrics/training_history.csv`
- `plots/train_loss_curve.png`
- `plots/val_loss_curve.png`
- `plots/val_dice_curve.png`
- `plots/val_iou_curve.png`
- `plots/learning_rate_curve.png`
- `plots/valid_cell_rate_curve.png`
- `plots/uncertain_cell_rate_curve.png`
- `overlays/epoch_XXX/*_ground_truth_overlay.png`
- `overlays/epoch_XXX/*_prediction_overlay.png`
- `overlays/epoch_XXX/*_error_overlay.png`

TensorBoard logs are written under:

```bash
tensorboard --logdir outputs/runs/<run_id>/tensorboard
```

Open the local HTML report with:

```bash
open outputs/runs/<run_id>/training_report.html
```

## Output Layout

Each run writes to:

```text
outputs/runs/YYYYMMDD_HHMMSS_experiment_name/
  configs/
  logs/
  checkpoints/
  predictions/
  overlays/
  csv/
  metrics/
  plots/
  dataset_preview/
  training_report.html
```

Important CSVs:

- `csv/dataset_summary.csv`
- `csv/train_manifest.csv`
- `csv/val_manifest.csv`
- `csv/matched_cells.csv`
- `csv/morphology_cell_level.csv`
- `csv/morphology_tile_summary.csv`
- `csv/morphology_case_summary.csv` when case IDs exist

## Metric Interpretation

- Train loss: should decrease without NaN or sudden explosion.
- Validation Dice/IoU: boundary overlap quality; inspect overlays even if
  these improve.
- Valid cell rate: fraction of boundaries containing exactly one nucleus and
  passing QC.
- Uncertain cell rate: cells rejected for zero/multiple nuclei, edge contact,
  stroma/vessel overlap, small size, or ambiguous labels.
- Edge rejection rate: high values usually indicate too many cut cells or
  tile-edge annotations entering measurement readouts.

Stop or pause training when validation metrics plateau and overlays show stable
measurement quality. Check annotations when valid cell rate is low, uncertain
rate is high, nuclei are systematically outside boundaries, compact/clear
labels are mixed, or false positives cluster in stroma/vessels.

## Legacy Notebook Workflow

The older Cellpose/rule-based prototype notes are kept below for reference.

This project keeps the step-by-step notebook workflow and adds a timestamped
pipeline runner plus a Streamlit QC app.

## Files

- `sample/tile_1488_x83811_y10131.png` - sample input image
- `cellpose_count_cells.py` - reusable Cellpose/image helper functions
- `cellpose_label_cells.ipynb` - step-by-step notebook for labeling cells
- `rule_based_histology_workflow.ipynb` - paper-style rule-based notebook
- `rule_based_histology_pipeline.py` - reusable rule-based H&E pipeline
- `app.py` - Streamlit QC interface
- `requirements.txt` - Python dependencies

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name cellpose-testing --display-name "Cellpose Testing"
```

## Notebook

Open `cellpose_label_cells.ipynb` and choose the `Cellpose Testing` kernel. The
notebook shows each stage separately:

1. Load the sample image.
2. Split H&E channels.
3. Preprocess nuclei/cytoplasm channels.
4. Detect and remove very dark pink lines plus large/elongated stroma.
5. Run Cellpose for nuclei on the cleaned nuclei channel.
6. Build residual cytoplasm labels as tissue minus nuclei and excluded stroma.
7. QC the filled cytoplasm overlay together with nuclei and stroma boundaries.
8. Filter and measure labels.
9. Export masks, overlays, CSV measurements, and YOLO segmentation labels.

## Color-Aware Pipeline

The production-style runner rejects pink stromal/background structures after
candidate generation. It combines:

- color exclusion masks for dark pink/magenta stroma,
- separate nuclei detection on hematoxylin,
- pale-green cell-body candidates,
- object-level filters for area, eccentricity, solidity, circularity,
  pink/magenta dominance, pale-green fraction, and nucleus matching.

Run it from Python:

```bash
python - <<'PY'
from cellpose_count_cells import CellposeConfig, find_sample_image, run_cell_segmentation_pipeline

result = run_cell_segmentation_pipeline(
    find_sample_image(),
    CellposeConfig(gpu=True, device="auto", log_timing=True),
    output_root="segmentation_runs",
)
print(result["summary"])
print(result["paths"]["run_dir"])
PY
```

Each run creates a new timestamped folder with:

- `overlay_cells_qc.png` - accepted cells green, rejected objects red, nuclei blue
- `cells.csv`, `accepted_cells.csv`, `rejected_objects.csv`
- `accepted_cells_qupath.geojson`, `rejected_objects_qupath.geojson`
- NumPy masks and `parameters.json`

## Rule-Based Histology Pipeline

`rule_based_histology_pipeline.py` is the color-first workflow for clear vs
compact cell analysis. It does not rely on Cellpose.

For single-image validation before WSI/cohort scaling, run:

```bash
python single_image_pathology_prototype.py sample/tile_1488_x83811_y10131.png
```

This writes a clean prototype output folder with `overlays/`, `csv/`, and
`summary/`. It also writes `qupath/single_image_annotations_qupath.geojson`
and `qupath/import_single_image_annotations.groovy` for QuPath import.

Current quantitative summaries use a compartment hierarchy:

1. Tissue/analysis area is normalized to 100%.
2. Parenchyme and mesenchyme are reported as percentages of tissue.
3. Nuclear and cytoplasmic areas are reported as percentages of parenchyme.
4. N/C ratio is nuclear area divided by cytoplasmic area.
5. Cell size is cytoplasmic region area plus assigned nuclear area.
6. Clear-cell and compact-cell components are reported as cytoplasm-area
   percentages, not as percentages of detected object count.

For step-by-step threshold tuning, open
`rule_based_histology_workflow.ipynb`. It shows the same stages inline before
running the exporter.

Stages:

1. Segment tissue into parenchyma and mesenchyme/stroma/background using H&E
   color features plus nucleus density.
2. Segment cytoplasmic cell regions first from vacuolated clear-cell
   compartments, eosin gradients, membrane-like ridges, and texture
   discontinuities. Weak membrane candidates use multi-scale gradients and
   suppress intracellular vacuoles so vacuole walls are not treated as cell
   borders.
3. Detect dark purple/blue nuclei only after cytoplasmic regions exist, then
   assign nuclei to the already-segmented cells. Large pale merged regions can
   be refined with nuclear spacing as a soft constraint, but the split is kept
   only when weak edge/texture support exists.
4. Classify cells by cytoplasm intensity/staining:
   clear cells are bright with low eosin/saturation; compact cells are darker
   or more eosinophilic.
5. Export QuPath GeoJSON, overlay PNGs, debug PNGs for each stage, and
   `cells.csv`.

Paper-style debug images are also saved:

- `debug/paper_stage1_parenchyma_red_mesenchyme_green.png`
- `debug/paper_stage2_nuclear_yellow_cytoplasm_non_yellow.png`
- `debug/stage2_detected_vacuolated_regions.png`
- `debug/stage2_inferred_membrane_boundaries.png`
- `debug/stage2_weak_membrane_boundaries.png`
- `debug/stage2_rejected_uncertain_regions.png`
- `debug/stage3_merged_region_candidates.png`
- `debug/stage3_soft_nuclear_refinement_boundaries.png`
- `debug/stage3_final_cell_polygons.png`
- `paper_stage3_clear_non_yellow_compact_yellow.png`

The main requested QC overlay remains
`overlay_clear_compact_mesenchyme.png`: clear cells green, compact cells blue,
mesenchyme red, nuclei boundaries yellow.

Run one tile:

```bash
python rule_based_histology_pipeline.py sample/tile_1488_x83811_y10131.png
```

For a large TIFF/WSI:

1. Open the TIFF in QuPath.
2. Run `qupath_export_tiles_for_rule_based_pipeline.groovy`.
3. Test a few tiles, then process all tiles:

```bash
python batch_process_rule_based_tiles.py "wsi_tiles/<image_name>" --limit 5
python batch_process_rule_based_tiles.py "wsi_tiles/<image_name>"
```

4. Reopen the TIFF in QuPath and run
   `qupath_import_rule_based_annotations.groovy`.

## Streamlit QC

```bash
streamlit run app.py --server.port 8507
```

The sidebar exposes Cellpose diameter, flow/cellprob thresholds, area, shape,
color, and nucleus-required filters. Every run is saved to a timestamped output
folder, so existing results are not overwritten.
