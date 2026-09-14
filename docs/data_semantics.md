# UrbanTrack AI — Data Semantics & Interface Contract (Phase 3)

## 1. Core Principles & Semantic Invariants

1. **Zero Fabrication**: If an observation modality is not captured by physical sensors, its value remains `None` (absent). Missing evidence is never synthesized or defaulted to positive values.
2. **Missing Evidence $\neq$ Negative Evidence**: Absence of license plate or OSNet embedding is neutral (`EVIDENCE_ABSENT`), whereas conflicting plates or impossible travel speeds are negative evidence (`EVIDENCE_CONTRADICTORY`).
3. **Coordinate System Rigor**: Single-camera detections exist in 2D image pixel space ($[x, y]$). They are never converted to GPS latitude/longitude or metric ground-plane distances without a verified homography matrix and camera calibration profile.
4. **Speed vs. Velocity**: Pixel displacement per second is recorded as `pixel_speed` ($\text{px/s}$). Physical speed in $\text{km/h}$ is strictly unavailable for single-camera video unless calibrated.
5. **Time Reference**: Timestamps produced from standalone video streams are `video_relative` ($\text{seconds from frame 0}$). They are never assumed to be synchronized with external cameras unless explicit clock synchronization metadata (`time_reference_id`, PTP/NTP) is documented.
6. **Probabilistic Honesty**: Match scores produced by heuristic combinations are `same_vehicle_score` and `identity_evidence_score` operating in $[0.0, 1.0]$. They are **decision ranking metrics**, NOT Bayesian posterior probabilities.

---

## 2. Comprehensive Field Dictionary

| Field Name | Producer | Semantic Meaning | Unit | Coordinate System | Time Reference | Aggregation Method | Missing-Value Semantics |
|---|---|---|---|---|---|---|---|
| `observation_id` | Member 2 Adapter | Unique identifier for the tracklet sighting | String UUID/Tag | N/A | N/A | Deterministic (`{cam}_trk_{id}`) | Never missing |
| `camera_id` | Member 1 Config | Identifier of the physical camera sensor | Alphanumeric | N/A | N/A | Preserved from stream | Never missing |
| `frame_id` | Member 1 Tracker | Video frame number where sighting starts | Integer | Discrete frames | Stream sequence | Sighting start frame | Integer $\ge 0$ |
| `timestamp_seconds` | Member 1 Adapter | Elapsed time from start of video stream | Seconds ($s$) | 1D time line | `video_relative` (or UTC if sync) | `frame_id / fps` | Float $\ge 0.0$ |
| `timestamp_semantics` | Member 1 / Sensor | Reference frame of timestamp | Enum | N/A | N/A | `"video_relative"` or `"synchronized"` | Default: `"video_relative"` |
| `time_reference_id` | Camera Metadata | Shared clock domain tag across sensors | String | N/A | N/A | Configured | `None` (independent clock) |
| `vehicle_type` | Member 1 YOLO | Coarse vehicle classification | String | Categorical | N/A | Plurality class vote across frames | `"car"` default fallback |
| `detection_confidence` | Member 1 YOLO | Mean confidence of vehicle bounding box detections | Unitless $[0.0, 1.0]$ | Probability proxy | Track interval | Arithmetic mean over track frames | `None` if undetected |
| `frame_detection_confidence_mean` | Member 1 Telemetry | Camera-wide mean detection confidence across all objects in frame | Unitless $[0.0, 1.0]$ | Frame quality | Single frame / window | Mean across camera detections | `None` if telemetry missing |
| `bbox` | Member 1 Tracker | Representative vehicle bounding box $[x_1, y_1, x_2, y_2]$ | Pixels ($\text{px}$) | Image ($3840 \times 2160$) | Frame snapshot | Median/Initial frame bbox | `None` if absent |
| `trajectory_point` | Member 1 Tracker | Representative spatial point of vehicle in image | Pixels $[x, y]$ | Image space | Frame snapshot | Mid-bottom of bounding box | `None` if bbox missing |
| `point_type` | Member 2 Adapter | Specific semantic of the image coordinate | String | Image space | N/A | `"image_space_trajectory_point"` | Must be documented |
| `point_coordinate_system` | Member 2 Adapter | Coordinate domain of the trajectory point | String | Image space | N/A | `"image"` (never `"ground"` without homography) | `"image"` |
| `pixel_speed` | Member 1 Tracker | Average 2D displacement speed in image plane | Pixels / sec ($\text{px/s}$) | Image space | Track lifespan | $\Delta \text{pixels} / \Delta t$ | `None` if track length $< 2$ |
| `appearance_embedding` | Member 1 OSNet | Deep feature vector representing vehicle visual appearance | 512-D float vector | L2 Unit Sphere | Feature space | Mean-pooled over tracklet crops, L2-normalized | `None` if crop unextractable |
| `plate` | Member 1 OCR | License plate alphanumeric string | Clean uppercase string | Text | Track lifespan | Multi-frame plurality consensus vote | `None` if unreadable / absent |
| `plate_confidence` | Member 1 OCR | OCR model recognition confidence | Unitless $[0.0, 1.0]$ | Model confidence | Consensus frames | Confidence of winning plate string | `None` if plate is None |
| `camera_reliability` | Member 1 Telemetry | Sensor trustworthiness / optical clarity | Unitless $[0.0, 1.0]$ | Quality scale | Track interval | Mean over track frames: $0.5 \cdot \text{rel} + 0.3 \cdot (1-\text{blur}) + 0.2 \cdot \text{bright}$ | Default: $0.85$ (uncalibrated) |
| `source_provenance` | Member 2 Adapter | Complete audit trace back to raw Member 1 files | Dictionary | N/A | N/A | Structural metadata mapping | Complete lineage preserved |

---

## 3. Mandatory Distinctions & Enforcements

### A. Timestamp Semantics
- When comparing two observations from **different cameras**:
  - If either observation has `timestamp_semantics = "video_relative"` and cameras lack a shared `time_reference_id`, temporal difference $\Delta t$ is **strictly non-comparable** (`status = "unavailable"`).
  - Temporal feasibility returns neutral evidence ($0.50$), and no physical travel speed is calculated.

### B. Image Coordinates vs. Physical World
- Trajectory coordinates extracted from single CCTV cameras are strictly `image_space_trajectory_point` in pixel units.
- They are **never converted to GPS coordinates** or metric meters without a calibrated homography matrix.
- Camera geographic location (`latitude`, `longitude`) represents the **camera sensor installation site**, NOT the physical coordinates of the vehicle.

### C. Local Track ID vs. Global Identity ID
- `track_id`: Camera-local tracker identifier assigned by ByteTrack/DeepSORT (e.g. `"65"`, `"94"`). Valid only within that specific camera stream.
- `identity_id`: Global vehicle identity cluster inferred by Member 2 across multiple tracklets and cameras (e.g. `"URB_00001"`).

### D. Decision States & Operating Thresholds
- `CONFIRMED`: $\text{same\_vehicle\_score} \ge 0.75$ with supporting multimodal evidence.
- `AMBIGUOUS`: $0.40 \le \text{same\_vehicle\_score} < 0.75$, or unconfirmed singletons lacking positive identity evidence.
- `REJECTED`: $\text{same\_vehicle\_score} < 0.40$, or physical contradiction (divergent plates, speed $> 120\text{ km/h}$, simultaneous presence).
