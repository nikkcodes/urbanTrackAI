# UrbanTrack AI — Layer 2 Canonical Input/Output Contract

**Contract Status**: **FROZEN & AUTHORITATIVE**  
**Date**: 2026-10-08  
**Scope**: Layer 2 Multi-Camera Intelligence, Association, and Global Identity Fusion across all 65 Scenario-Qualified Streams (S01–S06)  
**Input Reference**: Layer 1 Frozen Baseline ([UrbanTrack_Member1_Handoff 2](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202))  
**Audit Baseline**: [LAYER2_INPUT_AUDIT.md](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/LAYER2_INPUT_AUDIT.md)  
**Machine-Readable Schema**: [layer2_canonical_contract.json](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/layer2_canonical_contract.json)  

---

## Executive Summary & Design Scope

Following the completion of the Member 1 → Member 2 Layer 2 Input Boundary Audit (`LAYER2_INPUT_AUDIT = PASS`, Classification `B = READY WITH KNOWN LIMITATIONS`), this document defines and freezes the **Canonical Layer 2 Contract**.

Every component in the Layer 2 pipeline—including dataset loaders, temporal synchronizers, candidate pair generators, multi-modal evidence calculators, assignment solvers, and export modules—**must adhere strictly** to the data schemas, join protocols, validation invariants, and fallback policies defined herein.

No association algorithm, matching matrix, or identity solver is implemented in this document. This contract establishes the frozen boundary and mathematical invariants **before** writing Layer 2 inference code.

---

## 1. Canonical Tracklet Record

The **Canonical Tracklet Record** is the normalized, in-memory representation of a single camera-local vehicle tracklet produced by joining Layer 1 perception artifacts. It strictly distinguishes between camera-local identifiers and future global identities, and marks unavailable spatial or sensor quantities explicitly as `null`.

### 1.1 Structural Schema

A Canonical Tracklet is uniquely identified by the tuple `(scenario_id, camera_id, track_id)`.

```json
{
  "scenario_id": "S01",
  "camera_id": "CAM_S01_C001",
  "track_id": 1,
  "global_vehicle_id": null,
  "temporal": {
    "start_frame": 0,
    "end_frame": 51,
    "duration_frames": 52,
    "fps": 10.0,
    "start_raw_timestamp": "00:00:00.000",
    "end_raw_timestamp": "00:00:05.100",
    "start_raw_seconds": 0.000,
    "end_raw_seconds": 5.100,
    "start_sync_timestamp": "00:00:00.000",
    "end_sync_timestamp": "00:00:05.100",
    "start_sync_seconds": 0.000,
    "end_sync_seconds": 5.100,
    "duration_seconds": 5.100
  },
  "motion": {
    "trajectory_pixels": [[1135, 322], [1135, 321]],
    "trajectory_length": 42,
    "average_velocity_px": 1.60,
    "direction": "northbound"
  },
  "appearance": {
    "has_embedding": true,
    "appearance_embedding": [0.0312, -0.0145, "... 512 floats ..."],
    "embedding_dim": 512,
    "embedding_quality": 0.4914,
    "reid_model": "osnet_x0_25_aicity",
    "reid_compatibility_group": "AICITY_VEHICLE_V1"
  },
  "vehicle": {
    "vehicle_type": "car",
    "average_detector_confidence": 0.7946
  },
  "anpr": {
    "has_plate_detection": true,
    "has_readable_ocr": false,
    "aggregated_plate_text": null,
    "ocr_confidence": null,
    "ocr_readings_count": 0,
    "plate_detections_count": 14,
    "ocr_consensus_ratio": null
  },
  "spatial": {
    "camera_latitude": 42.525540,
    "camera_longitude": -90.723480,
    "camera_bearing_deg": 335.0,
    "camera_confidence": "HIGH",
    "road_context": {
      "road": "John F. Kennedy Road",
      "intersection": "Northwest Arterial & John F. Kennedy Road (Southeast corner)"
    },
    "projected_vehicle_coordinates": null
  },
  "quality": {
    "camera_reliability": 0.8150,
    "missing_evidence": ["READABLE_OCR"]
  }
}
```

### 1.2 Field Specification & Semantic Partitioning

1. **Identity Group**:
   - `scenario_id` (`str`): Operational scenario (`"S01"` through `"S06"`).
   - `camera_id` (`str`): Canonical scenario-qualified camera key (`"CAM_S01_C001"`).
   - `track_id` (`int`): ByteTrack tracking ID strictly positive and unique within `camera_id`.
   - `global_vehicle_id` (`str` or `null`): Multi-camera cross-scenario global identity. **MUST initially be `null`**. It is populated solely by the Layer 2 global association solver.

2. **Temporal Group**:
   - `start_frame` / `end_frame` (`int`): Video frame indices of first and last appearance.
   - `duration_frames` (`int`): $\text{end\_frame} - \text{start\_frame} + 1$.
   - `fps` (`float`): Camera capture frame rate (10.0 FPS across 64 cameras; 8.0 FPS for `CAM_S03_C015`).
   - `start_raw_timestamp` / `end_raw_timestamp` (`str`): Original video-relative `"HH:MM:SS.mmm"` string from `observations.json`.
   - `start_raw_seconds` / `end_raw_seconds` (`float`): Parsed video-relative float seconds from video start.
   - `start_sync_timestamp` / `end_sync_timestamp` (`str`): Synchronized timeline timestamp string formatted `"HH:MM:SS.mmm"`.
   - `start_sync_seconds` / `end_sync_seconds` (`float`): Scenario-synchronized timeline seconds ($t_\text{sync} = t_\text{raw} + \Delta t_\text{offset}$).
   - `duration_seconds` (`float`): Elapsed tracklet duration in seconds ($\text{end\_sync\_seconds} - \text{start\_sync\_seconds}$).

3. **Motion Group**:
   - `trajectory_pixels` (`list[list[int]]`): Sequence of $[x, y]$ bottom-center image sensor pixels.
   - `trajectory_length` (`int`): Count of trajectory sample coordinates.
   - `average_velocity_px` (`float`): Mean inter-frame pixel displacement.
   - `direction` (`str` or `null`): Predominant vehicle heading mode (`"northbound"`, `"southbound"`, `"eastbound"`, `"westbound"`, `"stationary"`, or `null`).

4. **Appearance Group**:
   - `has_embedding` (`bool`): `true` if valid 512-D vector exists; `false` if `appearance_embedding` is `null`.
   - `appearance_embedding` (`list[float]` or `null`): 512-D $L_2$-normalized float vector.
   - `embedding_dim` (`int` or `null`): Vector dimension (512 when present, `null` when missing).
   - `embedding_quality` (`float` or `null`): Detection confidence / image crop quality metric.
   - `reid_model` (`str`): Architecture string (`"osnet_x0_25_aicity"` or `"osnet_x0_25_msmt17"`).
   - `reid_compatibility_group` (`str`): Categorical compatibility domain:
     - `"AICITY_VEHICLE_V1"`: 64 cameras.
     - `"MSMT17_PERSON_BASELINE"`: 1 camera (`CAM_S01_C002`).
     - `"NONE"`: When embedding is missing.

5. **Vehicle Classification Group**:
   - `vehicle_type` (`str`): Object class (`"car"`, `"truck"`, `"bus"`, `"van"`).
   - `average_detector_confidence` (`float` or `null`): Mean YOLO detection confidence across observations.

6. **ANPR / OCR Group**:
   - `has_plate_detection` (`bool`): `true` if $\ge 1$ frame detected a plate bounding box.
   - `has_readable_ocr` (`bool`): `true` if $\ge 1$ frame produced a non-empty alphanumeric OCR string.
   - `aggregated_plate_text` (`str` or `null`): Normalized consensus alphanumeric plate string (e.g., `"IA789ABC"`).
   - `ocr_confidence` (`float` or `null`): Mean EasyOCR recognition confidence for consensus string.
   - `ocr_readings_count` (`int`): Count of successful OCR frames for this tracklet.
   - `plate_detections_count` (`int`): Count of frames with plate bounding boxes.
   - `ocr_consensus_ratio` (`float` or `null`): Fraction of readable frames matching consensus string.

7. **Spatial Group**:
   - `camera_latitude` / `camera_longitude` (`float`): Georeferenced static mounting position from [`camera_locations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_locations.json).
   - `camera_bearing_deg` (`float`): Optical viewing axis orientation in degrees $[0^\circ, 360^\circ)$.
   - `camera_confidence` (`str`): Georeferencing confidence (`"HIGH"` or `"MEDIUM"`).
   - `road_context` (`dict`): Human-readable road corridor and intersection names.
   - `projected_vehicle_coordinates` (`null`): **MUST BE `null`**. Ground-plane WGS84 vehicle positions are not pre-computed in Layer 1 JSON outputs.

8. **Quality Group**:
   - `camera_reliability` (`float`): Environmental score $[0.0, 1.0]$ from `perception_summary.json`.
   - `missing_evidence` (`list[str]`): Explicit list of missing modalities (`"APPEARANCE_EMBEDDING"`, `"READABLE_OCR"`, `"PLATE_DETECTION"`, `"MOTION_DIRECTION"`).

---

## 2. Observation → Tracklet Join Contract

To construct the Canonical Tracklet Record, Layer 2 must join `trajectories.json` with `observations.json`.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│        trajectories.json        │       │        observations.json        │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ - track_id (PK)                 │       │ - frames[].frame_number         │
│ - start_frame, end_frame        │ ◄───► │ - frames[].timestamp            │
│ - trajectory: [[x, y], ...]     │  JOIN │ - frames[].fps                  │
│ - appearance_embedding (512-D)  │  ON   │ - frames[].vehicles[].track_id  │
│ - embedding_quality             │       │ - frames[].vehicles[].bbox      │
│ - reid_model                    │ (cam, │ - frames[].vehicles[].direction │
│ - vehicle_type                  │ track)│ - frames[].vehicles[].plate_*   │
└─────────────────────────────────┘       └─────────────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │        Canonical Tracklet Record        │
                      └─────────────────────────────────────────┘
```

### 2.1 Join Key Definition
The join is performed per camera using the compound composite key:
$$\text{JoinKey} = (\text{camera\_id}, \text{track\_id})$$
where `camera_id` is inherited from the parent directory and `track_id` is the positive integer identifier.

### 2.2 Join Rules & Data Transformation Protocols

1. **Temporal Boundaries Validation**:
   - In `observations.json`, locate all frame records containing `vehicles` with `track_id == t.track_id`.
   - The minimum frame number in observations **must equal** `t.start_frame`.
   - The maximum frame number in observations **must equal** `t.end_frame`.
   - If observations are missing for an intermediate frame, the tracklet duration remains $[start\_frame, end\_frame]$ based on ByteTrack tracking continuity.

2. **Timestamp Recovery**:
   - `start_raw_timestamp` is extracted from the frame where `frame_number == t.start_frame`.
   - `end_raw_timestamp` is extracted from the frame where `frame_number == t.end_frame`.
   - Raw seconds are parsed from `"HH:MM:SS.mmm"`:
     $$t_\text{sec} = 3600 \cdot H + 60 \cdot M + S + \frac{m}{1000}$$

3. **Detector Confidence Aggregation**:
   - For all observations of `track_id`, extract `vehicle.confidence`.
   - Compute `average_detector_confidence = mean(confidences)`.

4. **Direction Aggregation**:
   - Filter observation records for non-null `direction` strings.
   - If available, select the statistical mode (majority direction). If all frames report `"stationary"` or are null, calculate displacement vector from `trajectory[0]` to `trajectory[-1]`.

5. **ANPR / License Plate Aggregation**:
   - Collect all observation frames where `plate_number` is not `null` and non-empty.
   - If count is 0:
     - `has_plate_detection` = `true` if `plate_bbox` was detected in any frame, else `false`.
     - `has_readable_ocr` = `false`.
     - `aggregated_plate_text` = `null`.
     - `ocr_confidence` = `null`.
   - If count $\ge 1$:
     - Normalize each raw string: uppercase alphanumeric characters only (`re.sub(r'[^A-Z0-9]', '', raw.upper())`).
     - Reject strings shorter than 3 characters as OCR noise.
     - Select majority vote string.
     - Compute `ocr_consensus_ratio = count(majority) / count(total_readable)`.
     - Compute `ocr_confidence = mean(plate_text_confidence)` for matching frames.

---

## 3. Timestamp Synchronization Contract

Layer 1 exports raw video-relative timestamps starting at `00:00:00.000` for each camera stream. Layer 2 **must compute** synchronized scenario timelines using official challenge metadata.

### 3.1 Mathematical Synchronization Formulation

For camera $C_i$ in scenario $S$:
$$t_\text{sync\_sec} = t_\text{raw\_sec} + \Delta t_\text{offset}(S, C_i)$$

Where:
- $t_\text{raw\_sec}$ is parsed from `observations.json` frame timestamp.
- $\Delta t_\text{offset}(S, C_i)$ is the authoritative camera clock offset in seconds from [`cam_timestamp/<SCENARIO>.txt`](file:///Users/yanalavivekreddy/Downloads/AICity22_Track1_MTMC_Tracking/cam_timestamp).

### 3.2 Authoritative Camera Offset Table

The official camera start offsets in seconds:

| Scenario | Camera Number | Offset $\Delta t$ (s) | Scenario | Camera Number | Offset $\Delta t$ (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | `c001` | 0.000 | **S04** | `c016` | 0.000 |
| **S01** | `c002` | 1.640 | **S04** | `c017` – `c020` | 14.318, 29.955, 26.979, 25.905 |
| **S01** | `c003` | 2.049 | **S04** | `c021` – `c025` | 39.973, 49.422, 45.716, 50.853, 50.263 |
| **S01** | `c004` | 2.177 | **S04** | `c026` – `c030` | 70.450, 85.097, 100.110, 125.788, 124.319 |
| **S01** | `c005` | 2.235 | **S04** | `c031` – `c035` | 125.033, 125.199, 150.893, 140.218, 165.568 |
| **S02** | `c006` | 0.000 | **S04** | `c036` – `c040` | 170.797, 170.567, 175.426, 175.644, 175.838 |
| **S02** | `c007` | 0.061 | **S05** | `c010`, `c016`–`c036` | **0.000** (all 19 cameras offset 0.0) |
| **S02** | `c008` | 0.421 | **S06** | `c041`–`c046` | **0.000** (all 6 cameras offset 0.0) |
| **S02** | `c009` | 0.660 | — | — | — |
| **S03** | `c010`–`c015` | `c010`: 8.715, `c011`: 8.457, `c012`: 5.879, `c013`: 0.0, `c014`: 5.042, `c015`: 8.492 |

### 3.3 Frame Rate Heterogeneity Contract
- **64 Cameras**: 10.0 FPS.
- **1 Camera**: **8.0 FPS** (`CAM_S03_C015`).
- **Rule**: Never assume a constant `0.1s` step size. Timestamps must be parsed directly from `observations.json` strings or calculated using `frame / fps` using camera-specific `fps`.

### 3.4 Direction-Independent Chronological Pairing Contract

> [!CAUTION]
> **Defect Prevention Rule**: Layer 2 must **NEVER** assume camera string ordering (`CAM_A < CAM_B`) matches physical vehicle travel direction. Assuming alphabetical order causes reverse-direction vehicles to compute negative $\Delta t$, resulting in false negative rejections.

For any candidate pair $(T_A, T_B)$:

1. **Establish Chronological Origin & Destination**:
   Compare the synchronized arrival/start times:
   $$\begin{cases}
   T_\text{origin} = T_A, \quad T_\text{dest} = T_B, \quad \text{Direction} = \text{"A\_TO\_B"} & \text{if } T_A.\text{start\_sync\_seconds} \le T_B.\text{start\_sync\_seconds} \\
   T_\text{origin} = T_B, \quad T_\text{dest} = T_A, \quad \text{Direction} = \text{"B\_TO\_A"} & \text{if } T_B.\text{start\_sync\_seconds} < T_A.\text{start\_sync\_seconds}
   \end{cases}$$

2. **Compute Physical Transit Time $\Delta t$**:
   Physical vehicle transit time $\Delta t$ is computed from departure at origin to arrival at destination:
   $$\Delta t = T_\text{dest}.\text{start\_sync\_seconds} - T_\text{origin}.\text{end\_sync\_seconds}$$

   *If cameras have overlapping FOVs (such that arrival occurs before departure), $\Delta t$ is computed as interval midpoint delta:*
   $$\Delta t_\text{mid} = T_\text{dest}.\text{mid\_sync\_seconds} - T_\text{origin}.\text{mid\_sync\_seconds}$$

3. **Enforce Invariant**:
   Under this chronological formulation, $\Delta t$ along the forward travel axis is **guaranteed to be non-negative** ($\Delta t \ge 0$). Negative time along the chosen direction is strictly prohibited.

---

## 4. Camera Graph Contract

The camera network transition topology is defined in [`camera_graph.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff%202/data/config/camera_graph.json).

### 4.1 Soft Topological Prior Semantics
- **Contract Rule**: `camera_graph.json` represents physical roadway connectivity. It is a **soft Bayesian prior**, **NEVER** a hard pruning filter.
- **When Directed Edge Exists $(C_\text{orig} \to C_\text{dest})$**:
  - Distance: $d = \text{edge.distance\_m}$
  - Forward Bearing: $\theta = \text{edge.bearing\_deg}$
  - Prior probability $P(C_\text{dest} \mid C_\text{orig})$ is elevated:
    $$\text{topology\_prior} = 0.80 + 0.15 \cdot \text{confidence\_weight}$$
- **When Directed Edge Does NOT Exist**:
  - Distance: $d = d_\text{haversine}(C_\text{orig}, C_\text{dest})$ computed from coordinates in `camera_locations.json`.
  - The candidate pair is **NOT** rejected.
  - Prior probability is baseline:
    $$\text{topology\_prior} = 0.20$$
  - Association remains governed by visual Re-ID, spatio-temporal velocity windowing, and ANPR agreement.

### 4.2 Directed Asymmetry Contract
- **Bidirectional Edges**: 168 edges (84 pairs). Traversal permitted in both directions.
- **One-Way Edges**: Exactly 2 edges in S02:
  1. `CAM_S02_C007 -> CAM_S02_C009` ($d = 30.1\text{m}$)
  2. `CAM_S02_C009 -> CAM_S02_C006` ($d = 29.1\text{m}$)
- Reverse traversal across one-way edges receives zero topological prior boost ($\text{topology\_prior} = 0.05$).

### 4.3 Scenario Isolation Contract
- Every edge in `camera_graph.json` satisfies:
  $$\text{scenario}(\text{edge.source}) == \text{scenario}(\text{edge.target})$$
- Cross-scenario graph edges are **strictly forbidden**. Cross-scenario transition prior is $0.0$.

---

## 5. Re-ID Compatibility Contract

Layer 1 contains two distinct Re-ID model tags across the 65 cameras:

```
┌────────────────────────────────────────────────────────┐
│             Re-ID Compatibility Domain                 │
├────────────────────────────┬───────────────────────────┤
│    AICITY_VEHICLE_V1       │   MSMT17_PERSON_BASELINE  │
│  (osnet_x0_25_aicity)      │   (osnet_x0_25_msmt17)    │
│         64 CAMERAS         │         1 CAMERA          │
│   (All S01 except C002;    │      (CAM_S01_C002)       │
│     All S02, S03, S04,     │                           │
│          S05, S06)         │                           │
└─────────────┬──────────────┴─────────────┬─────────────┘
              │                            │
              ▼                            ▼
  [ Cosine Similarity Valid ]    [ Incompatible Space ]
```

### 5.1 Compatibility Rules

1. **Intra-Domain Comparison (Permitted)**:
   When both tracklets belong to `"AICITY_VEHICLE_V1"`:
   $$S_\text{reid}(\vec{u}, \vec{v}) = \vec{u} \cdot \vec{v} \in [-1.0, 1.0]$$
   *(Since vectors are unit $L_2$-normalized, inner product equals cosine similarity).*

2. **Cross-Domain Comparison (Prohibited)**:
   When $\text{group}(T_A) \ne \text{group}(T_B)$ (specifically, matching `CAM_S01_C002` against any other camera):
   - Computing cosine similarity is **strictly prohibited**.
   - Output evidence structure:
     ```json
     {
       "status": "INCOMPATIBLE_SPACES",
       "similarity": null,
       "compatibility_group_a": "AICITY_VEHICLE_V1",
       "compatibility_group_b": "MSMT17_PERSON_BASELINE",
       "fallback_strategy": "NON_APPEARANCE_FUSION"
     }
     ```

### 5.2 Baseline Fallback Policy for `CAM_S01_C002`
When matching candidate pairs involving `CAM_S01_C002`:
- Re-ID appearance weight $w_\text{reid}$ is set to $0.0$.
- Association must be evaluated using:
  1. Spatio-temporal velocity windowing: $v = \frac{d}{\Delta t} \in [v_{\min}, v_{\max}]$
  2. Camera topology prior from `camera_graph.json`
  3. ANPR license plate alphanumeric matching (if readable OCR exists)
  4. Vehicle type classification agreement (`car` == `car`)
  5. Motion direction and velocity consistency

*(Note: The isolated Step 8 C002 AICity re-extraction experiment is an experimental artifact and must not be assumed in the frozen baseline contract).*

---

## 6. OCR / ANPR Contract

OCR evidence is sparse across the dataset (10.8% overall readability, 17 cameras with 0 readable text).

### 6.1 Status: Optional Confirmation Modality
- **Contract Invariant**: OCR is **never a mandatory gating requirement**. A pair lacking OCR **cannot** be discarded.
- Lack of OCR is a neutral event ($0$ penalty, $0$ boost).

### 6.2 String Normalization Protocol
Raw strings from `observations.json` must be normalized using:
```python
def normalize_plate(raw_text: str) -> str:
    # 1. Strip all whitespace, hyphens, and punctuation
    # 2. Convert to uppercase ASCII alphanumeric only
    return re.sub(r"[^A-Z0-9]", "", raw_text.strip().upper())
```

### 6.3 Tracklet-to-Tracklet OCR Scoring

Let $P_A$ and $P_B$ be the aggregated plate strings:

$$\text{ocr\_score}(P_A, P_B) = \begin{cases}
1.0 & \text{if } P_A == P_B \\
1.0 - \frac{\text{Levenshtein}(P_A, P_B)}{\max(|P_A|, |P_B|)} & \text{if } |P_A| \ge 3 \text{ and } |P_B| \ge 3 \\
\text{null} & \text{if } P_A \text{ is null or } P_B \text{ is null}
\end{cases}$$

- **High Agreement ($\text{ocr\_score} \ge 0.85$)**: Injects strong identity confirmation boost.
- **Strong Conflict ($\text{ocr\_score} < 0.40$ with high OCR confidence)**: Injects identity rejection penalty.
- **Missing OCR**: `ocr_score = null`, weight redistributed to appearance and spatio-temporal.

---

## 7. Missing-Evidence Contract

In real-world traffic surveillance, evidence is frequently missing:
- **Null Embeddings**: 868 tracklets (11.65%) have `appearance_embedding: null`.
- **Missing OCR**: 89.2% of detections have no readable text; 17 cameras have 0 text.
- **Missing Motion Heading**: Stationary vehicles or short tracks lack directional bearing.
- **Unavailable Vehicle GPS**: Bounding box tire-contact WGS84 positions are not pre-computed.

### 7.1 Principle: Preservation of Uncertainty
- **Core Axiom**: Absence of evidence is **NOT** evidence of absence.
- Missing evidence does **not** mean `DIFFERENT`, nor does it mean `SAME`.
- The system must dynamically adjust modality weights and record explicit `uncertainty_score`.

### 7.2 Dynamic Weight Redistribution
For candidate pair $(A, B)$ with active modalities $M \subseteq \{\text{appearance}, \text{spatiotemporal}, \text{topology}, \text{ocr}, \text{vehicletype}\}$:

$$w_m' = \frac{w_m}{\sum_{k \in M} w_k}, \quad \forall m \in M$$

If appearance is missing ($m \notin M$):
- Spatio-temporal and topology weights absorb the missing weight.
- `evidence_completeness` is penalized:
  $$\text{evidence\_completeness} = \sum_{m \in M} w_m^\text{nominal}$$
- Tracklet pairs with low `evidence_completeness` are flagged with `is_uncertain = true`.

---

## 8. Candidate Pair Contract

Every candidate link evaluated by Layer 2 must be represented as a normalized `CandidatePair` record.

### 8.1 Schema Specification

```json
{
  "scenario_id": "S01",
  "candidate_pair_id": "S01:CAM_S01_C001:1__CAM_S01_C003:5",
  "origin_tracklet_id": "CAM_S01_C001:1",
  "destination_tracklet_id": "CAM_S01_C003:5",
  "chronological_direction": "A_TO_B",
  "temporal": {
    "t_origin_departure_sync": 5.100,
    "t_dest_arrival_sync": 9.450,
    "delta_t_seconds": 4.350,
    "is_chronologically_valid": true,
    "temporal_feasibility": "FEASIBLE"
  },
  "spatial": {
    "camera_distance_m": 48.6,
    "implied_speed_mps": 11.17,
    "implied_speed_mph": 24.99,
    "speed_feasibility": "FEASIBLE"
  },
  "topology": {
    "relationship": "possible_transition",
    "has_graph_edge": true,
    "edge_distance_m": 48.6,
    "edge_bearing_deg": 312.4,
    "topology_prior": 0.85
  },
  "appearance_evidence": {
    "status": "COMPUTED",
    "compatibility": "AICITY_VEHICLE_V1",
    "cosine_similarity": 0.8842,
    "quality_origin": 0.4914,
    "quality_dest": 0.6210
  },
  "ocr_evidence": {
    "status": "MISSING_EVIDENCE",
    "plate_origin": null,
    "plate_dest": null,
    "ocr_score": null
  },
  "vehicle_type_evidence": {
    "type_origin": "car",
    "type_dest": "car",
    "is_type_match": true
  },
  "motion_evidence": {
    "direction_origin": "northbound",
    "direction_dest": "northbound",
    "heading_alignment_score": 0.92
  },
  "evidence_quality": {
    "completeness_ratio": 0.75,
    "missing_modalities": ["OCR"]
  }
}
```

---

## 9. Layer 2 Output Contract

Layer 2 exports multi-camera intelligence artifacts consumed by Member 3 (GIS Dashboard). Every decision must be fully auditable.

### 9.1 Output Artifact Inventory

1. [`identity_associations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/identity_associations.json): Audited pairwise match ledger.
2. [`journey_hypotheses.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/journey_hypotheses.json): Chained vehicle journeys over time.
3. [`multicam_intelligence.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/multicam_intelligence.json): Unified contract file for Member 3.
4. [`rejected_associations.json`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/results/rejected_associations.json): Negative candidate ledger with rejection reason codes.

### 9.2 Global Vehicle ID (`global_vehicle_id`) Convention
- Global identities are formatted:
  $$\text{global\_vehicle\_id} = \text{"GV\_<SCENARIO>\_<SEQUENCE:05d>"}$$
  *(e.g., `"GV_S01_00001"`, `"GV_S04_00142"`).*
- **Inviolable Rule**: `global_vehicle_id` **never overwrites** camera-local `track_id`.

### 9.3 Journey Hypothesis Schema
A vehicle journey consists of an ordered sequence of tracklets:

```json
{
  "global_vehicle_id": "GV_S01_00012",
  "scenario_id": "S01",
  "vehicle_type": "car",
  "consensus_plate": "IA789ABC",
  "total_cameras_visited": 3,
  "journey_start_sync": 12.450,
  "journey_end_sync": 48.900,
  "journey_duration_seconds": 36.450,
  "total_distance_m": 182.4,
  "average_corridor_speed_mph": 22.4,
  "confidence_score": 0.9120,
  "uncertainty_score": 0.0880,
  "path": [
    {
      "hop_index": 0,
      "camera_id": "CAM_S01_C001",
      "track_id": 14,
      "arrival_sync": 12.450,
      "departure_sync": 18.200,
      "direction": "northbound"
    },
    {
      "hop_index": 1,
      "camera_id": "CAM_S01_C003",
      "track_id": 9,
      "arrival_sync": 24.100,
      "departure_sync": 31.500,
      "transit_time_from_prev_s": 5.900,
      "transit_distance_from_prev_m": 48.6,
      "link_evidence_score": 0.9250
    }
  ]
}
```

### 9.4 Rejection Reason Taxonomy
Every rejected candidate pair must be logged in `rejected_associations.json` with an explicit reason code:
- `REJECT_CROSS_SCENARIO`: Pair spans different scenarios.
- `REJECT_NEGATIVE_CHRONOLOGICAL_TIME`: $\Delta t < 0$ along chosen forward axis.
- `REJECT_EXCESSIVE_SPEED`: Implied travel speed $v > v_{\max}$ (e.g., $> 75\text{ mph}$).
- `REJECT_INSUFFICIENT_SPEED`: Implied transit took longer than scenario max idle time ($> 15\text{ min}$).
- `REJECT_INCOMPATIBLE_REID_NO_FALLBACK`: Incompatible model spaces without sufficient spatio-temporal/OCR evidence.
- `REJECT_APPEARANCE_DIVERGENCE`: Cosine similarity below threshold ($< \tau_\text{reid}$).
- `REJECT_OCR_CONFLICT`: Distinct confirmed alphanumeric plates.
- `REJECT_HUNGARIAN_SUBOPTIMAL`: Valid candidate pruned by global Hungarian competitive assignment.

---

## 10. Machine-Checkable Invariants (Validation Contract)

Every Layer 2 component must enforce and validate these 12 invariants:

| Invariant Code | Invariant Name | Strict Assertion |
| :--- | :--- | :--- |
| **INV-01** | `NO_CROSS_SCENARIO_ASSOCIATION` | $\text{scenario}(T_A) == \text{scenario}(T_B)$ unconditionally. |
| **INV-02** | `NO_INCOMPATIBLE_REID_COMPARISON` | $\cos(\vec{u}_\text{msmt17}, \vec{v}_\text{aicity})$ is strictly forbidden. |
| **INV-03** | `NO_NEGATIVE_CHRONOLOGICAL_TIME` | $\Delta t = t_\text{dest,start} - t_\text{orig,end} \ge -\epsilon_\text{overlap}$ along travel axis. |
| **INV-04** | `NO_ALPHABETICAL_ORDERING_ASSUMPTION` | Travel direction is determined strictly by $t_\text{sync}$, never camera ID string order. |
| **INV-05** | `SYNCHRONIZED_TIMESTAMPS_EXPLICIT` | $t_\text{sync} = t_\text{raw} + \Delta t_\text{offset}$ using official `cam_timestamp` tables. |
| **INV-06** | `HETEROGENEOUS_FPS_HANDLED` | `CAM_S03_C015` evaluated at 8.0 FPS; other 64 cameras at 10.0 FPS. |
| **INV-07** | `MISSING_EMBEDDINGS_NON_FATAL` | Null embeddings trigger weight redistribution; never unhandled exceptions. |
| **INV-08** | `MISSING_OCR_NON_FATAL` | Missing license plate does not disqualify valid candidates. |
| **INV-09** | `NO_CROSS_SCENARIO_GRAPH_EDGES` | Graph lookups return 0 edges across distinct scenarios. |
| **INV-10** | `LOCAL_TRACK_ID_PRESERVATION` | Camera-local `track_id` is immutable; `global_vehicle_id` is a separate field. |
| **INV-11** | `AUDITABLE_EVIDENCE_RECORD` | Every accepted identity association includes an itemized multi-modal ledger. |
| **INV-12** | `AUDITABLE_REJECTION_RECORD` | Every rejected candidate pair is recorded with a taxonomy reason code. |

---

## 11. Known Layer 2 Limitations

The following empirical data limitations exist in the frozen Layer 1 baseline and must be accommodated by Layer 2:

1. **`CAM_S01_C002` Incompatible Baseline Re-ID**:
   In the frozen Layer 1 handoff, C002 uses `osnet_x0_25_msmt17` while all other 64 cameras use `osnet_x0_25_aicity`. Direct cosine similarity across C002 in S01 is impossible without re-extraction or fallback.
2. **Sparse License Plate Readability**:
   Only 10.8% of plate detections yield readable text, and 17 cameras have 0 readable plates. OCR must remain an opportunistic bonus.
3. **Absence of Precomputed Synchronized Timestamps**:
   Perception JSONs record only video-relative time. Layer 2 must ingest `cam_timestamp` tables and apply offsets dynamically.
4. **Absence of Precomputed Ground-Plane Vehicle GPS**:
   Vehicle detections and trajectories exist only in 2D sensor pixels. GIS distance windowing relies on static camera pole coordinates from `camera_locations.json`.

---

## 12. Final Verdict

The canonical input/output contract for Layer 2 is fully defined, mathematically grounded, and frozen.

### Verdict

**LAYER2_CANONICAL_CONTRACT = READY**
