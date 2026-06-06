# Dataset QC

| Tile | Image size | Mask size | Instances | Max label | Labels contiguous |
|---|---:|---:|---:|---:|---|
| p2_tile_01 | 513x512 | 513x512 | 19 | 19 | True |
| p2_tile_02 | 512x512 | 512x512 | 18 | 18 | True |
| p2_tile_03 | 512x512 | 512x512 | 25 | 25 | True |
| p2_tile_04 | 512x512 | 512x512 | 13 | 13 | True |
| p2_tile_05 | 512x512 | 512x512 | 19 | 19 | True |
| p2_tile_06 | 512x512 | 512x512 | 16 | 16 | True |
| p2_tile_07 | 512x518 | 512x518 | 21 | 21 | True |
| p2_tile_08 | 512x512 | 512x512 | 14 | 14 | True |
| p2_tile_09 | 512x512 | 512x512 | 16 | 16 | True |
| p2_tile_10 | 512x512 | 512x512 | 20 | 20 | True |
| p2_tile_11 | 512x512 | 512x512 | 21 | 21 | True |
| p2_tile_12 | 512x512 | 512x512 | 48 | 48 | True |
| p2_tile_13 | 512x512 | 512x512 | 33 | 33 | True |
| p2_tile_14 | 512x512 | 512x512 | 35 | 35 | True |
| p2_tile_15 | 512x512 | 512x512 | 23 | 23 | True |
| p2_tile_16 | 512x512 | 512x512 | 36 | 36 | True |
| p2_tile_17 | 512x512 | 512x512 | 18 | 18 | True |
| p2_tile_18 | 512x512 | 512x512 | 22 | 22 | True |
| p2_tile_19 | 512x512 | 512x512 | 23 | 23 | True |
| p2_tile_20 | 512x512 | 512x512 | 18 | 18 | True |
| yolo_tile_21 | 512x512 | 512x512 | 38 | 38 | True |
| yolo_tile_22 | 513x512 | 513x512 | 18 | 18 | True |
| yolo_tile_23 | 512x512 | 512x512 | 22 | 22 | True |
| yolo_tile_24 | 512x512 | 512x512 | 23 | 23 | True |
| yolo_tile_25 | 512x512 | 512x512 | 19 | 19 | True |
| yolo_tile_26 | 512x512 | 512x512 | 16 | 16 | True |
| yolo_tile_27 | 512x512 | 512x512 | 19 | 19 | True |
| yolo_tile_28 | 512x512 | 512x512 | 13 | 13 | True |
| yolo_tile_29 | 512x512 | 512x512 | 15 | 15 | True |
| yolo_tile_30 | 512x512 | 512x512 | 16 | 16 | True |
| yolo_tile_31 | 512x512 | 512x512 | 21 | 21 | True |
| yolo_tile_32 | 512x512 | 512x512 | 23 | 23 | True |
| yolo_tile_33 | 512x512 | 512x512 | 20 | 20 | True |

QC pass criteria: image/mask sizes match, labels are contiguous from 1..N, and instance counts match the export manifest.
