# UrbanTrack AI — Layer 2 Canonical Ingestion & Normalization Report

**Report Date**: 2026-10-08  
**Module**: `layer2.ingestion`  
**Execution Script**: [`run_layer2_ingestion.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/run_layer2_ingestion.py)  
**Contract Baseline**: [`LAYER2_CANONICAL_CONTRACT.md`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/LAYER2_CANONICAL_CONTRACT.md)  
**Generated Artifacts**:
- Canonical Tracklets: [`results/layer2_ingestion/canonical_tracklets.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/layer2_ingestion/canonical_tracklets.json) (54.8 MB)
- Ingestion Summary: [`results/layer2_ingestion/ingestion_summary.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/layer2_ingestion/ingestion_summary.json)
- Validation Report: [`results/layer2_ingestion/ingestion_validation.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/layer2_ingestion/ingestion_validation.json)

---

## 1. Executive Summary & What Was Implemented

We implemented the complete production **Layer 2 Ingestion and Normalization Package** ([`layer2/ingestion/`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/layer2/ingestion/)). The module ingests the frozen Layer 1 perception baseline across all 65 scenario-qualified camera streams, performs deterministic joins between trajectory tracklets and frame observations, applies official synchronization offsets, enforces direction-independent chronological ordering, aggregates sparse OCR evidence, evaluates Re-ID compatibility domains, and verifies machine-checkable invariants.

### Architecture Overview

```
layer2/
├── __init__.py
└── ingestion/
    ├── __init__.py                 # Public package interface
    ├── canonical_models.py         # CanonicalTracklet & Dataclass schemas
    ├── timestamp_sync.py           # Timestamp parser, synchronizer & chronological order
    ├── observation_join.py         # Invariant boundary join (trajectories + observations)
    ├── ocr_aggregation.py          # Alphanumeric normalization & majority voting
    ├── reid_compatibility.py       # Space compatibility rules (AICity vs MSMT17)
    ├── tracklet_loader.py          # Memory-efficient camera streaming coordinator
    └── validation.py               # Machine-checkable invariant validation suite
```

In accordance with user instructions, **no cross-camera matching, Hungarian assignment, identity fusion, or route chaining** was implemented in this task.

---

## 2. Exact Source Files Consumed

All inputs were consumed strictly from the frozen Layer 1 directory ([`UrbanTrack_Member1_Handoff 2`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202)) and official challenge synchronization metadata:

1. **Camera Stream Per-Camera Perception Outputs** (65 cameras):
   - [`observations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/observations.json): 65 files (761,935 frame-level vehicle detections).
   - [`trajectories.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/trajectories.json): 65 files (7,448 ByteTrack single-camera tracklets).
   - [`perception_summary.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/perception_summary.json): 65 files (camera reliability and frame rate).
   - [`camera_metrics.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/camera_metrics.json): 65 files.
2. **Central Spatial & Topology Configurations**:
   - [`camera_locations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_locations.json): Georeferenced WGS84 positions, bearings, and confidence ratings for all 65 cameras.
   - [`camera_graph.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_graph.json): 65 nodes and 170 directed transition edges.
   - [`aicity_manifest.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/aicity_manifest.json): Camera inventory linking video splits and calibration paths.
3. **Official Camera Clock Offsets**:
   - [`data/cityflowv2/cam_timestamp/`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/data/cityflowv2/cam_timestamp/): `S01.txt`, `S02.txt`, `S03.txt`, `S04.txt`, `S05.txt`, `S06.txt`.

*Zero files inside `UrbanTrack_Member1_Handoff 2/` were modified or overwritten.*

---

## 3. Tracklets Successfully Normalized

* **Expected Tracklets**: 7,448
* **Successfully Normalized Tracklets**: **7,448 (100.0%)**
* **Tracklet Failures**: **0**

### Scenario Breakdown

| Scenario | Cameras | Canonical Tracklets | % of Dataset |
| :---: | :---: | :---: | :---: |
| **S01** | 5 | 625 | 8.39% |
| **S02** | 4 | 730 | 9.80% |
| **S03** | 6 | 219 | 2.94% |
| **S04** | 25 | 781 | 10.49% |
| **S05** | 19 | 3,921 | 52.65% |
| **S06** | 6 | 1,172 | 15.74% |
| **Total** | **65** | **7,448** | **100.00%** |

Every normalized tracklet contains all 8 canonical semantic groups: `identity`, `temporal`, `motion`, `appearance`, `vehicle`, `anpr`, `spatial`, and `quality`.

---

## 4. Observations Successfully Joined

* **Total Observations Joined**: **761,935 (100.0%)**
* **Orphaned Observations**: **0**
* **Join Invariant Verification**:
  $$\min(\text{observation.frame\_number}) == \text{trajectory.start\_frame}$$
  $$\max(\text{observation.frame\_number}) == \text{trajectory.end\_frame}$$
  **Result**: 7,448 / 7,448 tracklets strictly satisfied the boundary invariant. Zero frame regressions or boundary divergences occurred.

---

## 5. Timestamp Synchronization Results

* **Official Offsets Ingestion**: 100% of cameras across all 6 scenarios had their official offsets loaded dynamically from `cam_timestamp/<SCENARIO>.txt`. Zero hardcoding was used.
* **Synchronized Formula Verified**:
  $$t_\text{sync} = t_\text{raw} + \Delta t_\text{offset}$$
* **Heterogeneous FPS Handled**:
  - `CAM_S03_C015`: Capture rate **8.0 FPS** correctly loaded and verified.
  - All other 64 cameras: Capture rate **10.0 FPS** correctly loaded and verified.
* **Direction-Independent Chronological Ordering**:
  Implemented in [`compare_chronological_order()`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/layer2/ingestion/timestamp_sync.py). The chronological origin is derived strictly from `start_sync_seconds`. Travel direction is never assumed to match camera string sorting.
* **Regression Test Passed**: A dedicated unit test reproducing the previous reverse-direction bug (`CAM_S01_C002` moving to `CAM_S01_C001`) passes unambiguously, proving that travel direction is independent of camera string ordering.

---

## 6. OCR Aggregation Statistics

* **Tracklets with Plate Bounding Box Detections**: **5,361 (71.98%)**
* **Tracklets with Stored Readable Alphanumeric Text**: **8 (0.11%)**
* **Tracklets without Stored Readable OCR**: **7,440 (99.89%)**
* **Key Finding on Layer 1 Data**:
  While `perception_summary.json` recorded 39,408 internal OCR attempts during perception pipeline runs, the actual serialized `observations.json` files contain non-null `plate_number` strings in exactly 118 frame observations corresponding to 8 tracklets.
* **Contract Compliance**:
  Missing OCR was safely flagged as `has_readable_ocr: false` with `aggregated_plate_text: null`. In accordance with `INV-08`, no tracklet was rejected, and missing OCR is not treated as negative identity evidence.

---

## 7. Re-ID Compatibility Statistics

Each tracklet was categorized into its canonical compatibility domain:

| Compatibility Group | Re-ID Model Tag | Tracklets | Valid Embeddings | Null Embeddings | Vector Dim | Compatibility Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **AICITY_VEHICLE_V1** | `osnet_x0_25_aicity` | 6,465 | 6,465 | 0 | 512 | Mutually compatible across 64 cameras |
| **MSMT17_PERSON_BASELINE**| `osnet_x0_25_msmt17` | 115 | 115 | 0 | 512 | Incompatible with AICity (`CAM_S01_C002`) |
| **NONE** | None / Null | 868 | 0 | 868 | null | Missing appearance vector |
| **Total** | — | **7,448** | **6,580** | **868** | — | — |

* All 6,580 valid embeddings are 512-dimensional, finite (0 NaN, 0 Inf), and unit $L_2$-normalized ($\|e\|_2 = 1.000 \pm 0.001$).
* Function [`are_reid_compatible()`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/layer2/ingestion/reid_compatibility.py) strictly returns `False` when comparing MSMT17 against AICity embeddings, preventing invalid cross-space cosine similarity.

---

## 8. Missing-Evidence Statistics

Missing evidence was explicitly preserved rather than fabricated:

* **Missing Appearance Embedding**: **868 tracklets (11.65%)** (`has_embedding: false`, `missing_evidence: ["APPEARANCE_EMBEDDING"]`).
* **Missing License Plate Detections**: **2,087 tracklets (28.02%)** (`has_plate_detection: false`).
* **Missing Readable OCR**: **7,440 tracklets (99.89%)** (`has_readable_ocr: false`).
* **Missing Motion Direction (Stationary)**: **1,412 tracklets (18.96%)** (`direction: "stationary"` or trajectory displacement $\le 2\text{px}$).
* **Precomputed Vehicle WGS84 Positions**: **7,448 tracklets (100.0% null)**. Correctly marked `null` as Layer 1 JSONs store only image pixel coordinates.

---

## 9. Machine-Checkable Invariant Validation Results

All 8 canonical invariants and structural integrity checks were verified:

| Invariant Code | Invariant Name | Evaluation Target | Violations | Status |
| :--- | :--- | :--- | :---: | :---: |
| **INV-01** | `SCENARIO_ISOLATION` | Scenario prefix matches camera ID prefix | 0 | **PASS** |
| **INV-02** | `NO_INCOMPATIBLE_REID_COMPARISON` | MSMT17 vs AICity cosine similarity prohibited | 0 | **PASS** |
| **INV-04** | `NO_ALPHABETICAL_TEMPORAL_ASSUMPTION` | Chronological origin derived from $t_\text{sync}$ | 0 | **PASS** |
| **INV-05** | `SYNCHRONIZED_TIMESTAMPS_EXPLICIT` | $t_\text{sync} = t_\text{raw} + \text{offset}$ finite floats | 0 | **PASS** |
| **INV-06** | `HETEROGENEOUS_FPS_HANDLED` | `CAM_S03_C015` at 8.0 FPS; 64 cams at 10.0 FPS | 0 | **PASS** |
| **INV-07** | `MISSING_EMBEDDINGS_NON_FATAL` | 868 null embeddings preserved without crash | 0 | **PASS** |
| **INV-08** | `MISSING_OCR_NON_FATAL` | 7,440 tracks without OCR preserved | 0 | **PASS** |
| **INV-10** | `LOCAL_TRACK_ID_PRESERVED` | `track_id` intact; `global_vehicle_id` null | 0 | **PASS** |
| **STRUCT-01** | `CAMERA_COVERAGE` | All 65 scenario cameras present | 0 | **PASS** |
| **STRUCT-02** | `TRACKLET_COVERAGE` | Exactly 7,448 tracklets produced | 0 | **PASS** |
| **STRUCT-03** | `KEY_UNIQUENESS` | Zero duplicate `(scenario, camera, track)` keys | 0 | **PASS** |
| **STRUCT-04** | `EMBEDDING_INTEGRITY` | 6,580 unit-norm 512-D finite vectors | 0 | **PASS** |

### Test Suite Execution
The automated test suite in [`tests/layer2/test_layer2_ingestion.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/tests/layer2/test_layer2_ingestion.py) executed 13 tests covering timestamp parsing, offset application, 8 FPS handling, OCR normalization, Re-ID compatibility, scenario isolation, duplicate detection, and reverse camera ordering regression:
```text
============================== 13 passed in 0.03s ==============================
```

---

## 10. Performance, Runtime & Resource Consumption

The ingestion pipeline employed single-camera streaming to prevent loading duplicate copies of the 761,935 frame observations into memory:

* **Total Runtime**: **7.37 seconds** (4.68s for loading/joining all 65 cameras, 2.69s for invariant validation and 54.8 MB JSON export).
* **Throughput**: **162,806 observations/second**; **1,591 tracklets/second**.
* **Peak Process Memory**: **755.1 MB** (well within standard workstation RAM limits).

---

## 11. Known Data Limitations Documented

1. **`CAM_S01_C002` Baseline Re-ID Incompatibility**: All 115 embeddings in C002 belong to `MSMT17_PERSON_BASELINE`. Direct cosine similarity across C002 in S01 remains disabled in the baseline.
2. **Extreme OCR Sparsity**: In the serialized Layer 1 perception JSONs, only 8 tracklets contain non-null plate strings. Future Layer 2 association will rely predominantly on spatio-temporal velocity windowing, topological priors, and visual Re-ID.
3. **Absence of Precomputed Vehicle WGS84 Positions**: `projected_vehicle_coordinates` is confirmed `null`. Spatial windowing relies on static camera pole coordinates from `camera_locations.json`.

---

## 12. Final Verdict

The Layer 2 Ingestion and Normalization module has processed all 65 scenario-qualified camera streams, 7,448 tracklets, and 761,935 observations without error, strictly satisfying all contract schemas and validation invariants.

### Verdict

**LAYER2_INGESTION = READY**
