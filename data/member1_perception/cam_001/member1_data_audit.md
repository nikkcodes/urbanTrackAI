# URBANTRACK AI — MEMBER 1 PERCEPTION ENGINE INTEGRATION AUDIT
**Phase 0: Technical Architecture & Authoritative Data Audit**
**Date:** September 14, 2026  
**Auditor:** Member 2 Lead Inference / Probabilistic Reasoning Engineer  
**Status:** COMPLETE (Zero fabricated values, strict data honesty)

---

## 1. Executive Summary

Member 1 (Kanishka) has provided the official perception outputs from the UrbanTrack AI vision stack (YOLOv8 + ByteTrack + OSNet Re-ID + Telemetry) processed on `traffics.mp4` (4K 3840x2160 @ 30.0 fps, 613 frames, 20.433 seconds duration).

The dataset consists of four authoritative files:
1. `observations.json` (also archived as `raw_frame_detections.json`): 150,848 lines, 4,821 frame-level vehicle detections across 613 frames.
2. `trajectories.json` (also archived as `track_embeddings.json`): 30,145 lines, 39 vehicle tracklets with 512-dimensional OSNet embeddings (`osnet_x0_25_msmt17`).
3. `camera_telemetry.json`: 6,138 lines, 613 frame-level sensor health records (blur, brightness, occlusion, vehicle density, reliability).
4. `video_summary.json`: 50 lines, global processing summary metrics.

All data integrity checks pass: **zero NaNs, zero infinities, 100% finite 512-dimensional embeddings, 100% track ID consistency between detections and tracklets, and 0 duplicate records**.

---

## 2. Phase 0 Audit Requirements (Items A through S)

### A. Existing Member 2 Input Format
Member 2 consumes standardized `Observation` dataclass instances defined in [`schemas/observation_schema.py`](schemas/observation_schema.py#L75-L145) and re-exported in [`inference/observation.py`](inference/observation.py#L1-L8).
- **Core identity fields**: `observation_id` (str), `camera_id` (str), `timestamp` (datetime), `timestamp_seconds` (float), `latitude` (Optional[float]), `longitude` (Optional[float]), `plate` (Optional[str]), `plate_confidence` (Optional[float]), `appearance_embedding` (Optional[List[float]]), `camera_reliability` (Optional[float]).
- **Perception extension fields**: `frame_id` (Optional[int]), `track_id` (Optional[str]), `vehicle_type` (Optional[str]), `detection_confidence` (Optional[float]), `bbox` (Optional[List[float]]), `trajectory_point` (Optional[List[float]]), `point_type` ("vehicle_footpoint"), `point_coordinate_system` ("image"), `pixel_speed` (Optional[float]), `plate_bbox` (Optional[List[float]]), `plate_text` (Optional[str]), `ocr_confidence` (Optional[float]), `local_track_history` (Optional[List[Any]]).
- **Temporal metadata fields**: `timestamp_semantics` ("video_relative" | "synchronized_utc"), `time_reference_id` (Optional[str]), `clock_offset_seconds` (Optional[float]), `time_uncertainty_seconds` (Optional[float]).

### B. Actual Member 1 File Structure
Member 1 outputs four JSON artifacts:
- **`observations.json`**: Top-level object containing:
  - `"camera_id"`: `"CAM_001"`
  - `"video_name"`: `"traffics.mp4"`
  - `"frames_processed"`: `613`
  - `"frames"`: Array of 613 frame objects (`frame_number`, `timestamp`, `fps`, `vehicle_count`, `vehicles`).
- **`trajectories.json`**: Top-level array of 39 tracklet summary dictionaries.
- **`camera_telemetry.json`**: Top-level object containing `"camera_id"`, `"video_name"`, `"frame_width"` (3840), `"frame_height"` (2160), `"frames_processed"` (613), and `"frames"` (array of 613 telemetry objects).
- **`video_summary.json`**: Top-level object with aggregate metrics (processing device `"cuda"`, duration `20.433s`, vehicle counts, OCR success metrics, camera reliability metrics).

### C. Actual Fields Present in Every File
1. **In `observations.json` (per vehicle detection)**:
   - `track_id` (int)
   - `embedding_id` (int)
   - `reid_model` (str, `"osnet_x0_25_msmt17"`)
   - `embedding_dim` (int, `512`)
   - `vehicle_type` (str: `"car"`, `"truck"`, `"motorcycle"`)
   - `confidence` (float, range [0.407, 0.942])
   - `bbox` ([x1, y1, x2, y2], integer pixel bounds)
   - `centroid` ([cx, cy], integer pixel coordinates)
   - `velocity_px` (float, pixel displacement)
   - `velocity_vector` ([dx, dy], integer/float pixel vector)
   - `direction` (str: `"eastbound"`, `"westbound"`, `"northbound"`, `"southbound"`, `"stationary"`)
   - `plate_bbox` (Optional[[x1, y1, x2, y2]])
   - `plate_confidence` (Optional[float], plate detector confidence)
   - `plate_number` (Optional[str], OCR character reading or null)
   - `plate_text_confidence` (Optional[float], OCR text confidence)
2. **In `trajectories.json` (per tracklet)**:
   - `track_id` (int)
   - `vehicle_type` (str)
   - `start_frame` (int)
   - `end_frame` (int)
   - `duration_frames` (int)
   - `trajectory` (List[[x, y]], centroid pixel history)
   - `trajectory_length` (int)
   - `average_velocity_px` (float)
   - `appearance_embedding` (List[float], exactly 512 floats)
   - `embedding_quality` (float, range [0.551, 0.942])
   - `embedding_dim` (int, `512`)
   - `reid_model` (str, `"osnet_x0_25_msmt17"`)
3. **In `camera_telemetry.json` (per frame)**:
   - `frame_number` (int, 0 to 612)
   - `timestamp` (str, `"HH:MM:SS.mmm"`)
   - `brightness` (float, [0.42, 0.49])
   - `blur_score` (float, [0.231, 0.509])
   - `occlusion_ratio` (float, [0.167, 0.333])
   - `vehicle_density` (str representing float, e.g. `"0.000001"`)
   - `detection_confidence_mean` (float, [0.776, 0.901])
   - `reliability` (float, [0.481, 0.559])

### D. Number of Observations
- Total frame-level detections: **4,821 observations** across 613 frames.
- Average detections per frame: **7.86 vehicles/frame** (peak: 12 vehicles in frame).

### E. Number of Unique Tracks
- Unique local track IDs: **39 tracks**.
- Track ID set: `[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 17, 18, 20, 21, 25, 26, 28, 34, 35, 40, 41, 46, 49, 52, 54, 56, 60, 65, 68, 73, 77, 78, 80, 83, 86, 87, 88, 94, 95]`.
- All 39 tracks appear in both `observations.json` and `trajectories.json` (100% intersection, 0 orphans).

### F. Number of Cameras
- Number of cameras in uploaded data: **1 camera (`CAM_001`)**.
- No physical second camera was provided in this video perception batch. (Cross-camera identity evaluations continue to be grounded via multi-camera synthetic/holdout benchmarks and camera-local re-entry analysis between Track 65 and Track 94).

### G. Number of Embeddings
- Number of Re-ID embeddings: **39 embeddings** (exactly one 512-dimensional vector per tracklet in `trajectories.json`).
- In `observations.json`, each frame detection points to its track's embedding via `embedding_id` (where `embedding_id == track_id`).

### H. Embedding Dimensions
- Declared dimension: **512**.
- Actual dimension verified: **512** for all 39 vectors.
- Numerical integrity: 100% finite floats; **0 NaNs, 0 Infinities**.

### I. Re-ID Model Identifier
- Model identifier: **`"osnet_x0_25_msmt17"`** (Omni-Scale Network, width 0.25, trained on MSMT17).

### J. Number of Valid vs Missing Plates
- Total frame-level detections: 4,821.
- Detections with plate bounding box detected (`plate_bbox` not null): **2,190 (45.43%)**.
- Detections with successful OCR string (`plate_number` not null and non-empty): **1,046 (21.70%)**.
- Detections with missing/unreadable plate (`plate_number` is null): **3,775 (78.30%)**.
- Unique physical plate strings identified across the video: **7 unique plates**:
  1. `NH01DP4248` (Track 1)
  2. `MA0AJK3437` (Track 4)
  3. `MH4JAD9203` (Track 6)
  4. `NH0LDD4922` (Track 8)
  5. `MH01BD1383` (Track 12 & 20)
  6. `MH0ZFX9484` (Track 65 & 94)
  7. `NH0LBD4932` (OCR variation of `NH0LDD4922`)

### K. OCR Confidence Availability
- Available on two levels:
  - `plate_confidence`: Available for all 2,190 plate bounding boxes (mean: 0.762).
  - `plate_text_confidence`: Available for all 1,046 recognized text strings (mean: 0.948).

### L. Camera Reliability Availability
- Available at frame level in `camera_telemetry.json` for all 613 frames.
- Range: `[0.4810, 0.5590]`, mean: `0.5164`.
- Grounded in physical sensor metrics: brightness ($0.45$), blur ($0.30$), occlusion ($0.24$), vehicle density ($10^{-6}$).

### M. Timestamp Format
- String format in observations: `"HH:MM:SS.mmm"` (e.g. `"00:00:00.000"` to `"00:00:20.400"`).
- Monotonically increasing at exactly $33.333\text{ ms}$ intervals.
- Semantics: **Video-relative elapsed time** from start of `traffics.mp4`. Not epoch/UTC wall-clock time.

### N. FPS Availability
- Explicitly provided in every frame record: `fps = 30.0`.

### O. Velocity Representation
- `velocity_px`: Scalar pixel displacement between consecutive frames (range `[0.0, 183.04]`, mean `4.92 px`).
- `velocity_vector`: 2D list `[dx, dy]` in pixel space.
- In `trajectories.json`: `average_velocity_px` per tracklet.
- **CRITICAL SEMANTICS:** This is purely **image pixel velocity**. No homography or ground-plane camera calibration was supplied. It MUST NEVER be converted to km/h or m/s without physical calibration.

### P. Camera Metadata Availability
- In perception JSON files: **None** (no latitude, longitude, road segment, or compass bearing).
- In Member 2 repository: [`data/cameras/camera_metadata.json`](data/cameras/camera_metadata.json) defines coordinates for `cam_01`, `cam_02`, `cam_03`.
- Inconsistency: Member 1 uses `"CAM_001"`, while repository metadata uses `"cam_01"`. The adapter must normalize camera ID casing and prefixes.

### Q. Inconsistencies Discovered
1. **Camera ID casing**: `CAM_001` vs `cam_01`.
2. **Vehicle density typing**: In `camera_telemetry.json`, `"vehicle_density"` is stored as a string (`"0.000001"`) rather than a float.
3. **Plate text variation**: Plate `NH0LDD4922` has occasional single-character OCR confusion (`NH0LBD4932`) on frames with heavy motion blur.
4. **Coordinate system**: Pixel coordinates range up to 3840x2160 (4K). Centroids and bounding boxes are image-plane, not GIS coordinates.

### R. Missing Fields Expected by Member 2
- `latitude` and `longitude`: Missing from raw perception files (must be attached from `CameraMetadata`).
- `timestamp_semantics`: Not declared in perception JSON (must be set to `"video_relative"` with `time_reference_id="CAM_001"`).
- `appearance_embedding` in `observations.json`: Stored per-tracklet in `trajectories.json` via `embedding_id`, not duplicated inside every frame detection (clean normalized storage).

### S. Duplicated or Malformed Records
- **Zero duplicates**: 0 duplicate frames, 0 duplicate tracks, 0 duplicate embedding IDs.
- **Zero malformed vectors**: All 39 vectors are exactly length 512, all numeric and finite.
- **Zero broken references**: All `embedding_id` and `track_id` cross-references resolve cleanly.

---

## 3. Data Semantics Contract (Phase 2)

| Dimension | Real Semantic Reality | Strict Member 2 Policy |
| :--- | :--- | :--- |
| **Timestamps** | Video-relative ($0.000\text{s} - 20.400\text{s}$) | Mark `timestamp_semantics = "video_relative"`. Never calculate cross-camera $\Delta t$ against UTC cameras without explicit clock offset. |
| **Camera Synchronization** | Single camera feed | Treat `CAM_001` as its own time reference frame (`time_reference_id = "CAM_001"`). |
| **Velocity** | Pixel displacement ($\text{px}/\text{frame}$) | Keep as `pixel_speed`. **NEVER convert to km/h** without homography matrix. |
| **Coordinates** | Image 2D $(x, y)$ in $[0, 3840] \times [0, 2160]$ | Store as `trajectory_point` with `point_coordinate_system = "image"`. NEVER infer GPS. |
| **Camera Reliability** | Real sensor telemetry ($0.481 - 0.559$) | Scales evidence trust weights. **NEVER multiplies physical traffic volume (PCU)**. |
| **Re-ID Quality** | Detection/mask confidence ($0.551 - 0.942$) | Modulates appearance evidence weight. Mismatched appearance on low-quality embeddings is AMBIGUOUS, not hard REJECT. |

---

## 4. Architectural Alignment Plan (Phases 3 - 20)

```
data/member1_perception/cam_001/
  ├── raw_frame_detections.json  (Authoritative raw observations)
  ├── track_embeddings.json      (Authoritative raw 512-d OSNet vectors)
  ├── camera_telemetry.json      (Authoritative frame telemetry)
  └── video_summary.json         (Authoritative run metadata)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ Member1ObservationAdapter                              │
│ • Case-insensitive camera ID normalization             │
│ • Tracklet-level character consensus OCR voting        │
│ • 512-d OSNet vector resolution via embedding_id       │
│ • Telemetry binding (blur, brightness, reliability)   │
│ • Strict semantic typing (video_relative, image coords)│
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ Member 2 Core Inference Engines                        │
│ • Re-ID Baseline Engine (Pure Cosine Nearest-Neighbor) │
│ • Multimodal Identity Fuser (Plate + Re-ID + ST Gate)  │
│ • Contradiction-Aware Identity Graph                   │
│ • Trajectory Reconstruction & Missing Camera Hypotheses│
│ • Member 3 Flow & Analytics Adapter                    │
└────────────────────────────────────────────────────────┘
```
