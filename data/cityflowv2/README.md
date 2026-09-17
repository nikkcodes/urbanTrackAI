# Extracted CityFlowV2 benchmark assets

These files were selectively extracted from:

`/Users/yanalavivekreddy/Downloads/AICity22_Track1_MTMC_Tracking.zip`

The reproduction benchmark uses the official `train/S01` native files:

- `cam_timestamp/S01.txt`, `cam_framenum/S01.txt`, `cam_loc/S01.png`
- `train/S01/c001` through `c005` calibration files
- each camera's `mtsc/mtsc_deepsort_mask_rcnn.txt` tracker output
- each camera's `gt/gt.txt` annotation file, read only by the evaluator
- `eval/ground_truth_train.txt`, `ReadMe.txt`, and `list_cam.txt`

The source video files (`train/S01/c*/vdo.avi`) were not extracted. The archive
contains approximately 16 GB compressed video, while the selected metadata,
tracklets, calibration, and annotations are approximately 8.4 MB. Video paths
remain in observation provenance so the run is explicit about this boundary.

Run the measured integration and all other checks from the project root with:

```bash
python3 scripts/reproduce_all.py
```

The CityFlow evaluator keeps ground-truth boxes in a separate evaluation data
structure. It never places target identity labels into `Observation` objects or
passes them to identity fusion.
