# UrbanTrack AI — Member 1 → Member 2 (Layer 2) Input Boundary Audit

**Audit Date**: 2026-10-08  
**Audit Target**: `UrbanTrack_Member1_Handoff 2` (Layer 1 Perception, Data Processing, Validation & GIS Foundation)  
**Consumer Target**: Member 2 (Layer 2 Multi-Camera Intelligence, Spatio-Temporal Association & Identity Fusion)  
**Authoritative Reference**: [MEMBER1_HANDOFF.md](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/MEMBER1_HANDOFF.md)  
**Machine-Readable Dataset**: [layer2_input_audit.json](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/layer2_input_audit.json)  

---

## Executive Summary

This audit evaluates the frozen **Layer 1 perception package** across all **65 scenario-qualified video streams** to determine whether it provides the necessary inputs for implementing **Layer 2 (Multi-Camera Intelligence & Identity Fusion)**.

### Core Verdict
- **Structural Integrity**: **100% PASS** (260/260 required camera JSON files present, 0 missing files, 0 schema mismatches, 0 duplicate tracks, 0 invalid bounding boxes, 0 NaN/Inf embeddings).
- **Scale**: **65 Cameras**, **7,448 Tracklets**, **761,935 Observations**, **128,538 Frames**, **6,580 Valid 512-D Embeddings**, **364,968 Plate Detections**.
- **Critical Limitations Identified**:
  1. **Re-ID Embedding Incompatibility in S01**: `CAM_S01_C002` was extracted using `osnet_x0_25_msmt17` (person Re-ID weights), whereas all other 64 cameras used `osnet_x0_25_aicity` (vehicle Re-ID weights). Naive cosine similarity across C002 in S01 is mathematically invalid in the baseline handoff.
  2. **Absence of Synchronized Timestamps**: All timestamps in `observations.json` are video-relative strings (`HH:MM:SS.mmm`). Synchronized timestamps are not provided in Layer 1 outputs and must be computed by Layer 2 using scenario clock offsets.
  3. **Trajectography Schema Boundary**: `trajectories.json` provides single-camera tracklet bounding-box trajectories in pixel space, but contains **no timestamps** (only frame indices) and **no OCR information**. Layer 2 must ingest and join `trajectories.json` with `observations.json`.
  4. **Sparse OCR Coverage**: Only **10.80%** of plate detections yield readable alphanumeric text (39,408 successes / 364,968 attempts), and **17 out of 65 cameras** have **0 readable plates**. OCR cannot serve as a hard requirement.
  5. **Null Embedding Rate**: **868 tracks (11.65%)** across 39 cameras lack appearance embeddings (`appearance_embedding: null`), necessitating spatio-temporal fallback association.

**Final Audit Classification**: **`B = READY WITH KNOWN LIMITATIONS`**  
**Audit Status**: **`LAYER2_INPUT_AUDIT = PASS`**

---

## 1. Camera Inventory Verification

We verified the directory inventory of [UrbanTrack_Member1_Handoff 2/data/output](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output) against the AI City Challenge 2022 Track 1 MTMC specification and [aicity_manifest.json](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/aicity_manifest.json).

### 1.1 Scenario Distribution
The dataset contains exactly **65 scenario-qualified video streams** partitioned across 6 operational scenarios:

| Scenario | Camera Count | Physical Roadway Region in Dubuque, IA | Camera IDs Included |
| :---: | :---: | :--- | :--- |
| **S01** | **5** | Northwest Arterial (IA-32) & John F. Kennedy Rd | `CAM_S01_C001` – `CAM_S01_C005` |
| **S02** | **4** | US-20 (Dodge St) & Century Dr | `CAM_S02_C006` – `CAM_S02_C009` |
| **S03** | **6** | Historic Hillside Corridor (Hill St / W 5th / Alpine) | `CAM_S03_C010` – `CAM_S03_C015` |
| **S04** | **25** | University Ave Corridor (Flora Park to Downtown) | `CAM_S04_C016` – `CAM_S04_C040` |
| **S05** | **19** | University Ave & Hill St Corridor Subset | `CAM_S05_C010`, `CAM_S05_C016`–`C029`, `CAM_S05_C033`–`C036` |
| **S06** | **6** | Expressway Arterial Progression along US-20 | `CAM_S06_C041` – `CAM_S06_C046` |
| **Total** | **65** | **Dubuque, IA Metropolitan Traffic Grid** | **65 Unique Scenario-Qualified Streams** |

### 1.2 Scenario-Qualified Identifier Uniqueness
- **Total Directory Entries**: 65
- **Unique Scenario-Qualified Keys**: 65 (100% uniqueness)
- **Format**: Strictly conforms to `CAM_<SCENARIO>_<CAMERA>` (`CAM_S[0-9]{2}_C[0-9]{3}`).

### 1.3 Prevention of Numeric Camera ID Collisions
There are **46 physical numeric camera numbers** (`C001` through `C046`) distributed over 65 streams. Numeric IDs are reused across scenarios because identical physical intersections were recorded during different sessions/scenarios:
- `C010` is recorded in both **S03** (`CAM_S03_C010`) and **S05** (`CAM_S05_C010`).
- `C016`–`C029` (14 cameras) are recorded in both **S04** and **S05**.
- `C033`–`C036` (4 cameras) are recorded in both **S04** and **S05**.

**Integrity Check**:
- Numeric camera IDs are **strictly separated** by scenario qualification.
- No directory, configuration node, or graph edge conflates cameras across scenarios.
- S03 and S05 instances of `C010` monitor the same physical intersection (Hill St & W 5th St), but maintain distinct metadata, observation counts, and trajectory files.

---

## 2. Required Files & Schema Verification

For every one of the 65 cameras, we audited the presence and non-emptiness of the 4 required output artifacts under `UrbanTrack_Member1_Handoff 2/data/output/<CAMERA_ID>/`:

| Required File | Expected Count | Found Count | Missing Count | Status | Schema Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| [`observations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/observations.json) | 65 | 65 | 0 | **PASS** | Frame-by-frame vehicle detections, bounding boxes, velocities, and OCR |
| [`trajectories.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/trajectories.json) | 65 | 65 | 0 | **PASS** | ByteTrack single-camera tracklets, frame intervals, and 512-D Re-ID embeddings |
| [`perception_summary.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/perception_summary.json) | 65 | 65 | 0 | **PASS** | Per-camera perception metrics, track counts, OCR success rates, reliability |
| [`camera_metrics.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/output/CAM_S01_C001/camera_metrics.json) | 65 | 65 | 0 | **PASS** | Frame-level environmental quality: brightness, blur, occlusion, reliability |

### Configuration Artifacts Verification
We verified the presence and valid JSON structure of all 3 central configuration files under [UrbanTrack_Member1_Handoff 2/data/config/](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/):

1. [`camera_graph.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_graph.json): 47,646 bytes, valid JSON (65 nodes, 170 directed transition edges).
2. [`camera_locations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_locations.json): 47,182 bytes, valid JSON (65 georeferenced camera locations with coordinates, bearing, and road context).
3. [`aicity_manifest.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/aicity_manifest.json): 35,216 bytes, valid JSON (65 camera entries linking video files, calibrations, and ground truth).

---

## 3. `trajectories.json` Schema & Field Audit

We audited all **7,448 tracklet records** across all 65 `trajectories.json` files.

### 3.1 Field Availability & Null Rates
Every tracklet record across all 65 cameras strictly adheres to a uniform 12-key schema:

```json
{
  "track_id": 1,
  "vehicle_type": "car",
  "start_frame": 0,
  "end_frame": 51,
  "duration_frames": 52,
  "trajectory": [[1135, 322], "..."],
  "trajectory_length": 42,
  "average_velocity_px": 1.6,
  "appearance_embedding": [0.0312, "... 512 floats ..."],
  "embedding_quality": 0.4914,
  "embedding_dim": 512,
  "reid_model": "osnet_x0_25_aicity"
}
```

Detailed field audit across all 7,448 tracks:

| Field Name | Type | Key Presence | Non-Null Count | Null Count | Null Rate | Semantic Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `track_id` | integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Single-camera ByteTrack track identifier ($\ge 1$) |
| `vehicle_type` | string | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | YOLO vehicle classification (`car`, `truck`, `bus`) |
| `start_frame` | integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Video frame number of first detection ($\ge 0$) |
| `end_frame` | integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Video frame number of last detection ($\ge \text{start}$) |
| `duration_frames` | integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | $\text{end\_frame} - \text{start\_frame} + 1$ ($\ge 1$) |
| `trajectory` | list | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Sequence of $[x, y]$ bottom-center pixel coordinates |
| `trajectory_length`| integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Number of points in trajectory path |
| `average_velocity_px` | float | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Average displacement per frame in sensor pixels |
| `appearance_embedding`| list | 7,448 / 7,448 (100%) | 6,580 | 868 | **11.65%** | 512-D $L_2$-normalized OSNet appearance feature vector |
| `embedding_quality` | float | 7,448 / 7,448 (100%) | 6,580 | 868 | **11.65%** | Crop detection confidence / image quality metric |
| `embedding_dim` | integer | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Feature vector dimensionality (always `512`) |
| `reid_model` | string | 7,448 / 7,448 (100%) | 7,448 | 0 | **0.00%** | Architecture tag: `osnet_x0_25_aicity` or `msmt17` |

### 3.2 Critical Schema Deficiencies in `trajectories.json`
By code and schema inspection, the following expected multi-camera tracking fields are **completely absent** from `trajectories.json`:
1. **Start / End Timestamps**: `start_timestamp` and `end_timestamp` do **NOT** exist in `trajectories.json`. Only frame numbers (`start_frame`, `end_frame`) are recorded.
2. **Motion Direction / Bearing**: Compass heading or 2D velocity vector ($\vec{v}$) is **NOT** present in `trajectories.json` (only scalar pixel speed `average_velocity_px`).
3. **License Plate / OCR**: `plate_number`, `ocr_text`, and plate confidence are **NOT** propagated to `trajectories.json`.
4. **Detection Confidence**: Only `embedding_quality` is recorded; detector classification confidence is omitted.

> [!IMPORTANT]
> **Layer 2 Requirement**: To perform temporal windowing and license plate matching, Layer 2 cannot rely on `trajectories.json` in isolation. Layer 2 **must join** `trajectories.json` with `observations.json` on `(camera_id, track_id)`.

---

## 4. Re-ID Embedding Space & Compatibility Audit

We audited all 7,448 tracklets across all 65 cameras for Re-ID model tag, embedding dimensions, and vector properties.

### 4.1 Model Inventory & Compatibility Groups

| Compatibility Group | Model Tag | Cameras | Total Tracks | Valid Embeddings | Null Embeddings | Null Rate | Vector Dim | Compatibility Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Group 1 (AICity Vehicle)** | `osnet_x0_25_aicity` | **64** | 7,323 | 6,465 | 858 | 11.72% | 512 | **MUTUALLY COMPATIBLE** |
| **Group 2 (MSMT17 Person)** | `osnet_x0_25_msmt17` | **1** (`CAM_S01_C002`) | 125 | 115 | 10 | 8.00% | 512 | **INCOMPATIBLE WITH GROUP 1** |
| **Total** | — | **65** | **7,448** | **6,580** | **868** | **11.65%** | **512** | **Split Embedding Space in S01** |

### 4.2 Mathematical Incompatibility of `CAM_S01_C002`
- In `UrbanTrack_Member1_Handoff 2/data/output/CAM_S01_C002/trajectories.json`, all 125 tracks list `reid_model: "osnet_x0_25_msmt17"`.
- All other cameras in S01 (`CAM_S01_C001`, `C003`, `C004`, `C005`) and all cameras across S02–S06 list `reid_model: "osnet_x0_25_aicity"`.
- **Mathematical Implication**: Even though both models output 512-D $L_2$-normalized vectors, MSMT17 was trained on human pedestrians, whereas AICity was fine-tuned on vehicles. Computing cosine similarity:
  $$\cos(\vec{u}_\text{msmt17}, \vec{v}_\text{aicity}) = \vec{u} \cdot \vec{v}$$
  produces arbitrary, degraded similarity values (~0.45–0.50 for true positives) with zero discriminative power.
- **Audited Provenance**: In the Step 9 experiment, comparing MSMT17 vs AICity embeddings for the identical 115 C002 vehicle crops showed a mean cosine of 0.4907 and standard deviation 0.0898, with 0/115 tracks reaching $\ge 0.90$.

> [!WARNING]
> In the frozen Member 1 handoff, `CAM_S01_C002` **cannot** be matched with other S01 cameras using appearance cosine similarity. Layer 2 must either use the isolated C002 AICity re-extraction experiment or fall back to spatio-temporal/OCR association for C002.

---

## 5. Temporal Data & Synchronization Audit

We audited the timestamp representations across all 128,538 frames in `observations.json` and all `perception_summary.json` files.

### 5.1 Timestamp Fields Present in Layer 1
1. **Frame Number**: Present as `frame_number` in `observations.json` ($0 \le f \le N-1$) and `start_frame`/`end_frame` in `trajectories.json`. (100% available).
2. **Raw Video-Relative Timestamp**: Present as string formatted `HH:MM:SS.mmm` (e.g. `"00:00:00.000"`, `"00:03:15.500"`). Strictly increases monotonically with frame number. (100% available).
3. **Frame Rate (FPS)**:
   - **64 Cameras**: 10.0 FPS.
   - **1 Camera**: **8.0 FPS** (`CAM_S03_C015`).

### 5.2 Absence of Synchronized Timestamps
- **Total Cameras with Synchronized Timestamps**: **0 / 65 (0.0%)**
- **Total Frames with Synchronized Timestamps**: **0 / 128,538 (0.0%)**
- Neither `observations.json`, `trajectories.json`, nor `perception_summary.json` contains synchronized timestamps or scenario clock offsets.
- **Layer 2 Requirement**: Layer 2 must load external scenario start offset metadata (`cam_timestamp/<SCENARIO>.txt`) and compute the synchronized timeline:
  $$t_\text{sync} = t_\text{raw} + \Delta t_\text{cam\_offset}$$
  taking into account the 8.0 FPS rate of `CAM_S03_C015`.

---

## 6. Spatial & GIS Data Audit

We audited spatial representations across `camera_locations.json`, `camera_graph.json`, and perception files.

### 6.1 Static Camera Georeferencing
In [`camera_locations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_locations.json), all 65 cameras have:
- **WGS84 Coordinates**: `latitude` $[42.4919, 42.5256]^\circ\text{N}$, `longitude` $[-90.7237, -90.6864]^\circ\text{W}$ (Dubuque, Iowa).
- **Road & Intersection Context**: Documented intersection names (e.g., *Northwest Arterial & John F. Kennedy Rd*).
- **Viewing Orientation**: Optical axis bearing in degrees $[0^\circ, 360^\circ)$.
- **Confidence Rating**: **61 HIGH**, **4 MEDIUM** (`C024`, `C029` in S04/S05), **0 LOW**.

### 6.2 Ground-Plane Homography Matrices
- Planar homography matrices are **NOT** stored inside `UrbanTrack_Member1_Handoff 2`.
- `aicity_manifest.json` provides relative paths pointing to `calibration.txt` in the external raw dataset root (`AICity22_Track1_MTMC_Tracking`).
- The utility script [UrbanTrack_Member1_Handoff 2/scripts/project_to_gis.py](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/scripts/project_to_gis.py) provides the $H^{-1}$ inverse mapping from bounding box bottom-center pixel $(u, v)$ to $(lat, lon)$.

### 6.3 Vehicle Observation Coordinates
- Vehicle positions in `observations.json` and `trajectories.json` are **strictly 2D pixel coordinates** on the image sensor:
  - `bbox`: $[x_1, y_1, x_2, y_2]$ in pixels $[0, W] \times [0, H]$
  - `centroid`: $[x, y]$ in pixels
  - `trajectory`: list of $[x, y]$ pixels
- Projected WGS84 coordinates are **not pre-computed** in Layer 1 JSON files.

---

## 7. Camera Graph Topology Audit

We audited the directed transition network in [`camera_graph.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_graph.json).

### 7.1 Graph Metrics & Structure
- **Nodes**: Exactly **65 scenario-qualified nodes** (1:1 correspondence with `camera_locations.json`).
- **Edges**: Exactly **170 directed edges**.
- **Edge Reciprocity**:
  - **168 Bidirectional Edges** (84 symmetric pairs).
  - **2 One-Way Directed Edges**:
    1. `CAM_S02_C007 -> CAM_S02_C009` (distance: 30.1m, US-20 / Century Dr eastbound channelized movement)
    2. `CAM_S02_C009 -> CAM_S02_C006` (distance: 29.1m, US-20 / Century Dr westbound movement)
- **Edge Distance Range**:
  - Min: **7.3 meters** (adjacent intersection views)
  - Max: **1,256.4 meters** (expressway corridor segments in S06)
  - Mean: **179.7 meters**
- **Forward Bearings**: Range from $1.3^\circ$ to $359.4^\circ$.
- **Edge Semantics**: All 170 edges specify `"relationship": "possible_transition"`.
- **Cross-Scenario Edges**: **0** (strictly 0 cross-scenario edges; scenarios are mutually partitioned).

### 7.2 Edges by Scenario

| Scenario | Active Cameras | Directed Edges | Topological Structure |
| :---: | :---: | :---: | :--- |
| **S01** | 5 | 18 | Full 4-way signalized intersection movements (NW Arterial & JFK Rd) |
| **S02** | 4 | 10 | Divided highway signal with 2 one-way turn lanes (US-20 & Century Dr) |
| **S03** | 6 | 14 | Historic steep hillside cluster (Hill St / W 5th / Alpine) |
| **S04** | 25 | 66 | Arterial corridor (~2.8 km along University Ave) |
| **S05** | 19 | 50 | University Ave & Hill St corridor subset (fully connected component) |
| **S06** | 6 | 12 | Linear progression along US-20 expressway |
| **Total** | **65** | **170** | **Physical Roadway Transition Corridors** |

> [!NOTE]
> As specified in [CAMERA_GRAPH.md](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/docs/CAMERA_GRAPH.md), the camera graph represents **physical road layout and candidate corridors**. It is **not** ground-truth vehicle movement and must serve as a **soft Bayesian prior**, never as a hard candidate rejection filter.

---

## 8. License Plate Detection & OCR Audit

We audited plate detections, OCR attempts, and recognized text across all 65 cameras.

### 8.1 Summary OCR Statistics
- **Total Plate Detections**: **364,968**
- **Total OCR Extraction Attempts**: **364,968**
- **Total OCR Successes (Readable Alphanumeric Text)**: **39,408**
- **Total OCR Failures (Unreadable / Blur / Glare)**: **325,560**
- **Overall OCR Success Rate**: **10.80%**
- **Cameras with Readable OCR ($\ge 1$ success)**: **48 Cameras (73.8%)**
- **Cameras with Plates Detected but ZERO Readable Text**: **17 Cameras (26.2%)**
- **Cameras with Zero Plate Detections**: **0 Cameras**

### 8.2 The 17 Cameras with Zero Readable OCR
The following 17 cameras detected thousands of license plate patches, but yielded **zero** readable text strings due to camera resolution, vehicle distance, angle, or speed:

| Camera ID | Scenario | Plate Detections | OCR Successes | Reason / Camera Context |
| :--- | :---: | :---: | :---: | :--- |
| `CAM_S02_C009` | S02 | 6,698 | 0 | Long-range divided highway approach on Dodge St |
| `CAM_S03_C010` | S03 | 4,038 | 0 | Steep angle downward view on Hill St |
| `CAM_S03_C011` | S03 | 3,877 | 0 | Hillside intersection with tree canopy shadow |
| `CAM_S04_C017` | S04 | 1,283 | 0 | Distant westbound approach on University Ave |
| `CAM_S04_C018` | S04 | 1,057 | 0 | Distant approach view |
| `CAM_S04_C019` | S04 | 176 | 0 | Low traffic mast arm view |
| `CAM_S04_C020` | S04 | 1,896 | 0 | Wide corridor view |
| `CAM_S04_C024` | S04 | 1,726 | 0 | Multi-leg intersection approach |
| `CAM_S04_C025` | S04 | 1,628 | 0 | High-speed University Ave section |
| `CAM_S04_C033` | S04 | 2,469 | 0 | Asbury Rd split |
| `CAM_S04_C035` | S04 | 103 | 0 | Low-volume approach |
| `CAM_S05_C023` | S05 | 3,422 | 0 | Long focal length corridor |
| `CAM_S05_C035` | S05 | 3,528 | 0 | Overhead mast arm |
| `CAM_S06_C041` | S06 | 2,807 | 0 | Expressway speed (US-20 Center Grove) |
| `CAM_S06_C042` | S06 | 661 | 0 | Expressway speed (US-20 Cedar Cross Rd) |
| `CAM_S06_C045` | S06 | 3,709 | 0 | Expressway speed (US-20 Walmart junction) |
| `CAM_S06_C046` | S06 | 1,797 | 0 | Expressway speed (US-20 Old Hwy Rd) |

### 8.3 Implications for Layer 2
- In S06 (expressway), out of 13,035 detected plates, only **2** yielded readable text across the entire scenario (0.015% success rate).
- License plate matching **cannot** be a required gate for cross-camera association.
- Layer 2 must treat ANPR as an **opportunistic confirmation bonus** when valid alphanumeric strings match.

---

## 9. Data Integrity & Anomaly Audit

We ran automated integrity checks across all 65 cameras.

| Integrity Check | Target Artifact | Audit Result | Defect Count | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Duplicate Track IDs** | `trajectories.json` per camera | Checked 7,448 tracks | 0 duplicates | **PASS** |
| **Duplicate Vehicle IDs** | `observations.json` per frame | Checked 128,538 frames | 0 duplicates | **PASS** |
| **Timestamp Format & Monotonicity**| `observations.json` frames | Checked 128,538 frames | 0 invalid | **PASS** |
| **Bounding Box Validity** ($w>0, h>0$)| `observations.json` detections | Checked 761,935 bboxes | 0 inverted/empty | **PASS** |
| **Embedding NaN / Inf Check** | `trajectories.json` embeddings | Checked 6,580 vectors | 0 NaN or Inf | **PASS** |
| **Embedding Dimension Uniformity** | `trajectories.json` embeddings | Checked 6,580 vectors | 0 non-512 | **PASS** |
| **Embedding $L_2$ Normalization** | `trajectories.json` embeddings | $\|e\|_2 = 1.000 \pm 0.001$ | 0 unnormalized | **PASS** |
| **Zero-Norm Vectors** | `trajectories.json` embeddings | Checked 6,580 vectors | 0 zero vectors | **PASS** |
| **Missing Files** | Output directories | Checked 260 files | 0 missing | **PASS** |
| **Cross-Camera Schema Consistency**| All 65 cameras | Uniform keys across files | 0 discrepancies | **PASS** |

The data exhibits **outstanding mechanical integrity**. There are no corrupted records, malformed JSON structures, or out-of-range numerical values.

---

## 10. Scale & Per-Camera Metrics

### 10.1 Global Dataset Scale

| Metric | Total Count across 65 Streams |
| :--- | :---: |
| **Total Cameras** | **65** |
| **Total Tracklets** | **7,448** |
| **Total Detections / Observations** | **761,935** |
| **Total Video Frames** | **128,538** |
| **Total Re-ID Embeddings** | **6,580** |
| **Total Plate Detections** | **364,968** |
| **Total Readable OCR Text Instances** | **39,408** |

### 10.2 Scale Breakdown by Scenario

| Scenario | Cameras | Tracklets | Observations | Frames Processed | Re-ID Embeddings | Null Embeddings | License Plates | Readable OCR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | 5 | 625 | 41,888 | 10,281 | 491 | 134 | 18,978 | 1,509 |
| **S02** | 4 | 730 | 35,268 | 8,109 | 618 | 112 | 16,724 | 1,404 |
| **S03** | 6 | 219 | 62,048 | 13,517 | 211 | 8 | 37,990 | 8,487 |
| **S04** | 25 | 781 | 79,482 | 10,780 | 728 | 53 | 40,102 | 2,694 |
| **S05** | 19 | 3,921 | 476,569 | 73,845 | 3,741 | 180 | 238,139 | 25,312 |
| **S06** | 6 | 1,172 | 66,680 | 12,006 | 791 | 381 | 13,035 | 2 |
| **Total** | **65** | **7,448** | **761,935** | **128,538** | **6,580** | **868** | **364,968** | **39,408** |

### 10.3 Extrema & Camera Extremes
- **Largest Camera by Observations**: `CAM_S05_C022` (47,046 observations, 153 tracks, 4,277 frames)
- **Largest Camera by Tracks**: `CAM_S05_C033` (384 tracks, 40,854 observations, 3,407 frames)
- **Smallest Camera by Tracks**: `CAM_S04_C019` and `CAM_S04_C024` (15 tracks each)
- **Largest Scenario**: **S05** accounts for **62.5% of all vehicle observations** (476,569 / 761,935) and **52.6% of all tracklets** (3,921 / 7,448).

---

## 11. Layer 1 vs. Layer 2 Architectural Boundary

To prevent architectural bleeding and misplaced responsibilities, the boundary between Layer 1 and Layer 2 is explicitly partitioned:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AVAILABLE FROM LAYER 1 (FROZEN)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 65 Scenario-Qualified Output Directories (CAM_S01_C001 ... CAM_S06_C046) │
│ 2. Single-camera ByteTrack tracklets (track_id, start_frame, end_frame)     │
│ 3. 2D Frame Detections (bbox [x1, y1, x2, y2], centroid, velocity_px)       │
│ 4. 512-D L2-normalized OSNet appearance embeddings (88.35% track coverage)  │
│ 5. Frame-level license plate detections and EasyOCR strings (observations)  │
│ 6. Per-frame camera environmental reliability and image metrics             │
│ 7. Static georeferenced camera coordinates and bearings (camera_locations)  │
│ 8. Directed physical transition network of 170 edges (camera_graph)         │
│ 9. Homography GIS projection utility (scripts/project_to_gis.py)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MUST BE IMPLEMENTED BY LAYER 2                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Multi-Camera Synchronization:                                            │
│    Ingest cam_timestamp offsets and map raw timestamps to unified timeline  │
│ 2. Heterogeneous FPS Handling:                                              │
│    Normalize frame-to-time conversion (handling 8.0 FPS CAM_S03_C015)       │
│ 3. Trajectory-Observation Data Join:                                        │
│    Join trajectories.json with observations.json for timestamps & ANPR      │
│ 4. Tracklet ANPR / OCR Aggregation:                                         │
│    Aggregate frame plate readings into tracklet-level consensus strings     │
│ 5. Spatio-Temporal Velocity & Direction Gating:                             │
│    Direction-aware delta_t = t_dest - t_orig > 0 and v = dist / delta_t     │
│ 6. Soft Topological Prior Integration:                                      │
│    Incorporate camera_graph transition distances without hard pruning       │
│ 7. Cross-Camera Appearance Association:                                     │
│    Cosine similarity over compatible 512-D AICity embedding space           │
│ 8. C002 Embedding Incompatibility Fallback:                                 │
│    Handle CAM_S01_C002 MSMT17 weights in S01 baseline                       │
│ 9. Multi-Modal Identity Fusion & Global Graph Matching:                     │
│    Weighted fusion (appearance + spatio-temporal + topology + OCR) & global │
│    Hungarian / Linear Assignment to produce global_vehicle_id journeys      │
│ 10. Scenario Partitioning:                                                  │
│    Enforce strict isolation between S01–S06 (zero cross-scenario matching)  │
│ 11. Packaging Layer 2 Intelligence Outputs:                                 │
│    Emit multicam_intelligence.json and unified journey graphs for Member 3  │
└─────────────────────────────────────────────────────────────────────────────┘
```

*(Note: Layer 3 responsibilities—such as Leaflet map visualization, React UI, and real-time dashboard endpoints—remain strictly outside Layer 2).*

---

## 12. Critical Blockers & Known Limitations Analysis

We investigated the 6 potential risks that could threaten scientifically valid 65-camera tracking:

### 12.1 Incompatible Re-ID Embedding Spaces (LIMITATION)
- **Status**: **KNOWN LIMITATION in S01 Baseline**.
- **Details**: 64 cameras share `osnet_x0_25_aicity`, while `CAM_S01_C002` uses `osnet_x0_25_msmt17`.
- **Impact**: Pairwise cosine similarity involving `CAM_S01_C002` in S01 fails under the frozen baseline.
- **Resolution for Layer 2**: Layer 2 can either use the isolated C002 AICity re-extraction experiment (Step 8) or utilize spatio-temporal and OCR modalities for C002 transitions. Scenarios S02, S03, S04, S05, and S06 are unaffected (100% pure AICity).

### 12.2 Missing Synchronized Timestamps (ARCHITECTURAL REQUIREMENT)
- **Status**: **LAYER 2 RESPONSIBILITY**.
- **Details**: Layer 1 outputs contain raw video timestamps (`HH:MM:SS.mmm`). Synchronized timestamps are omitted.
- **Resolution for Layer 2**: Layer 2 must load official `cam_timestamp/<SCENARIO>.txt` offset tables and adjust timestamps.

### 12.3 Trajectory Schema Deficiencies (INTEGRATION REQUIREMENT)
- **Status**: **RESOLVABLE VIA JOIN**.
- **Details**: `trajectories.json` omits timestamps and license plate text.
- **Resolution for Layer 2**: Layer 2 must load both `trajectories.json` and `observations.json` and join records on `track_id`.

### 12.4 Sparse OCR Availability (MODALITY CHARACTERISTIC)
- **Status**: **KNOWN DATA CHARACTERISTIC**.
- **Details**: Only 10.8% of plates are readable; 17 cameras have 0 readable plates; S06 has only 2 readable plates.
- **Resolution for Layer 2**: Layer 2 must treat ANPR as a soft confirmation bonus rather than a hard gating requirement.

### 12.5 Missing Calibration Files in Handoff Package (NON-BLOCKER)
- **Status**: **NON-BLOCKER**.
- **Details**: `calibration.txt` homographies are not duplicated in `UrbanTrack_Member1_Handoff 2`, but camera GPS coordinates and pairwise transition distances are already pre-computed in `camera_locations.json` and `camera_graph.json`.
- **Resolution for Layer 2**: Camera-to-camera spatial distance windowing does not require raw homographies.

### 12.6 Scenario Mixing Risks (CONTROLLED)
- **Status**: **CONTROLLED**.
- **Details**: Layer 1 strictly adheres to canonical scenario-qualified keys (`CAM_S01_C001` through `CAM_S06_C046`).
- **Resolution for Layer 2**: Layer 2 must maintain scenario partitioning by processing each scenario as an independent tracking domain.

---

## 13. Audit Conclusion & Verdict

The Layer 1 perception and GIS foundation provides a mechanically flawless, schema-consistent, and comprehensive baseline across all 65 scenario-qualified camera streams in Dubuque, Iowa. All 260 per-camera output files exist, 0 records are corrupted, and 64 out of 65 cameras share the fine-tuned 512-D AICity Re-ID embedding space.

The identified limitations (`CAM_S01_C002` MSMT17 embedding tag, raw video-relative timestamps, missing timestamps in `trajectories.json`, and sparse OCR) are well-characterized data characteristics that define the exact contract Layer 2 must implement.

---

### Classification

**B = READY WITH KNOWN LIMITATIONS**

---

### Audit Verification Status

**LAYER2_INPUT_AUDIT = PASS**
