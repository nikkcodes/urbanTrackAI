# UrbanTrack AI — Member 2 → Member 3 Interface Contract

**Document Version**: 1.0.0  
**Specification Level**: Normative Contract  
**Producer**: Member 2 (Probabilistic Vehicle Intelligence Engine)  
**Consumer**: Member 3 (City Mobility Analytics, Simulation, GIS & Dashboard)  
**Base Artifacts**: `UrbanTrack_Member2_Handoff/outputs/`  
**JSON Schemas**: `UrbanTrack_Member2_Handoff/schemas/`  

---

## 1. Overview and Contractual Principles

This document defines the formal data exchange contract between Member 2 and Member 3. Member 3 applications (such as OD matrix estimation, trajectory visualization, traffic simulation, and incident investigation) must consume these structures strictly according to the types, semantics, units, and nullability constraints defined herein.

### Core Contract Rules:
1. **Never Reconstruct Member 2 Logic**: Member 3 must consume inferred identities and graph components directly; do not re-cluster tracklets or re-evaluate Re-ID cosine distances.
2. **Respect Nullability as Conservative Uncertainty**: Fields such as `confidence` are explicitly `null` for unconfirmed singletons. Member 3 must never treat `null` as `0.0` or fabricate a default confidence.
3. **Coordinate System Rigor**: All world coordinates are metric planar coordinates in the local CityFlow benchmark coordinate frame (`cityflow_world`). Units are meters. They are **NOT** WGS84 GPS latitude/longitude.
4. **Timestamp Semantics**: Timestamps are measured in elapsed seconds from camera synchronization base. They are **NOT** UTC wall-clock timestamps.
5. **Separation of Inference from Evaluation**: Ground-truth identifiers (`gt_vehicle_id`) are evaluation-only and must never appear in production analytics or operational pipelines.

---

## 2. Inferred Identity Contract (`inferred_identities.json`)

File: `UrbanTrack_Member2_Handoff/outputs/inferred_identities.json`  
Schema: `UrbanTrack_Member2_Handoff/schemas/inferred_identity_schema.json`  
Top-level Structure: Array of `InferredIdentity` objects (355 entries in current build).

| Field Name | Type | Meaning & Semantics | Units | Allowed Values | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `identity_id` | `string` | Unique inferred global identity identifier | N/A | Regex: `^UT_ID_[0-9]{4}$` | No | Member 2 | Inferred | `"UT_ID_0002"` |
| `identity_status` | `string` | Cluster formulation level | N/A | `"candidate"`, `"unconfirmed_singleton"` | No | Member 2 | Inferred | `"candidate"` |
| `admission_status` | `string` | Physical consistency admission status | N/A | `"confirmed"`, `"unconfirmed_singleton"`, `"ambiguous"`, `"rejected_merge"` | No | Member 2 | Inferred | `"confirmed"` |
| `cameras` | `array[string]` | Distinct camera IDs where vehicle was observed | N/A | Elements: `"CAM_S01_C001"`, `"CAM_S01_C002"`, `"CAM_S01_C003"` | No | Member 2 | Inferred | `["CAM_S01_C001"]` |
| `observation_ids` | `array[string]` | Member 1 tracklet IDs belonging to this cluster | N/A | List of valid tracklet IDs | No | Member 1 / Member 2 | Inferred linkage | `["CAM_S01_C001_track_1", "CAM_S01_C001_track_5"]` |
| `observations_count` | `integer` | Count of tracklet observations in cluster | integer count | $\ge 1$ | No | Member 2 | Inferred | `2` |
| `vehicle_type` | `string` | Consensus vehicle classification across tracklets | N/A | `"car"`, `"truck"`, `"bus"`, `"suv"`, `"vehicle"`, `null` | Yes | Member 1 / Member 2 | Inferred consensus | `"car"` |
| `confidence` | `number` | Platt-calibrated probabilistic match score | probability in $[0.0, 1.0]$ | $0.0 \le p \le 1.0$ | Yes (`null` for singletons) | Member 2 | Inferred (Calibrated) | `0.7061` |
| `consistency.is_consistent` | `boolean` | Overall physical feasibility of cluster | N/A | `true`, `false` | No | Member 2 | Inferred | `true` |
| `consistency.status` | `string` | Detailed physical consistency classification | N/A | `"consistent"`, `"inconsistent"`, `"insufficient_evidence"`, `"unverified"`, `"singleton"` | No | Member 2 | Inferred | `"consistent"` |
| `consistency.consistency_score` | `number` | Quantitative physical feasibility metric | $[0.0, 1.0]$ | $0.0 \le s \le 1.0$ | Yes | Member 2 | Inferred | `1.0` |
| `consistency.temporal_consistency`| `string` | Non-overlapping temporal admissibility | N/A | `"consistent"`, `"inconsistent"`, `"unavailable"`, `"unverified"` | No | Member 2 | Inferred | `"consistent"` |
| `consistency.spatial_consistency` | `string` | Velocity and transition admissibility | N/A | `"consistent"`, `"inconsistent"`, `"unavailable"` | No | Member 2 | Inferred | `"consistent"` |
| `consistency.vehicle_type_consistency` | `string` | Type compatibility across tracklets | N/A | `"compatible"`, `"incompatible"`, `"unknown"` | No | Member 2 | Inferred | `"compatible"` |
| `consistency.violations` | `array[string]` | Physical violation flags detected | N/A | String descriptions of violations | No | Member 2 | Inferred | `[]` |
| `evidence_summary.edge_count` | `integer` | Confirmed pairwise edges within cluster | integer count | $\ge 0$ | No | Member 2 | Inferred | `1` |
| `evidence_summary.mean_edge_weight`| `number` | Average heuristic fusion weight of internal edges | $[0.0, 1.0]$ | $0.0 \le w \le 1.0$ | Yes | Member 2 | Inferred | `0.8523` |
| `evidence_summary.min_edge_weight` | `number` | Minimum heuristic edge weight in cluster | $[0.0, 1.0]$ | $0.0 \le w \le 1.0$ | Yes | Member 2 | Inferred | `0.8523` |
| `evidence_summary.modal_explanation`| `string` | Primary physical justification for identity link | N/A | Free text explanation | Yes | Member 2 | Inferred | `"High visual appearance similarity (0.852); Spatiotemporally feasible"` |
| `metadata.provenance` | `string` | Lineage tracing string | N/A | Structured provenance descriptor | No | Member 2 | Lineage | `"Member 2 IdentityGraph via Member 1 Tracklets"` |
| `metadata.cluster_algorithm` | `string` | Graph partitioning method | N/A | `"connected_components_with_transitive_audit"` | No | Member 2 | Metadata | `"connected_components_with_transitive_audit"` |
| `metadata.calibration_applied` | `boolean` | Indicates if Platt calibration was applied | N/A | `true`, `false` | No | Member 2 | Metadata | `true` |

---

## 3. Trajectory Contract (`trajectories.json`)

File: `UrbanTrack_Member2_Handoff/outputs/trajectories.json`  
Schema: `UrbanTrack_Member2_Handoff/schemas/trajectory_schema.json`  
Top-level Structure: Array of `VehicleTrajectory` objects (355 entries).

| Field Name | Type | Meaning & Semantics | Units | Allowed Values | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `identity_id` | `string` | Identity hypothesis this trajectory represents | N/A | Matches `InferredIdentity.identity_id` | No | Member 2 | Inferred | `"UT_ID_0002"` |
| `observations_count` | `integer` | Number of sequential waypoints/tracklets | count | $\ge 1$ | No | Member 2 | Inferred | `2` |
| `cameras_visited` | `array[string]` | Chronological sequence of visited camera IDs | N/A | Valid camera identifiers | No | Member 2 | Inferred | `["CAM_S01_C001"]` |
| `start_timestamp` | `number` | Journey start time (synchronized frame time) | seconds | $\ge 0.0$ | No | Member 1 / AI City | Inferred / Sync | `0.0` |
| `end_timestamp` | `number` | Journey end time (synchronized frame time) | seconds | $\ge \text{start\_timestamp}$ | No | Member 1 / AI City | Inferred / Sync | `18.5` |
| `total_time_seconds` | `number` | Total duration spanned by trajectory | seconds | $\ge 0.0$ | No | Member 2 | Inferred | `18.5` |
| `total_distance_m` | `number` | Traversed path distance | meters | $\ge 0.0$ | No | Member 2 | Inferred | `0.0` (unassociated road) |
| `overall_confidence` | `number` | Kinematic and observation confidence | $[0.0, 1.0]$ | $0.0 \le c \le 1.0$ | No | Member 2 | Inferred | `0.85` |
| `is_ambiguous` | `boolean` | Flag indicating route or association ambiguity | N/A | `true`, `false` | No | Member 2 | Inferred | `false` |
| `most_likely_route` | `array` | Sequence of road graph edge IDs | N/A | List of segment IDs or empty | No | Member 2 | Inferred | `[]` |
| `complete_route_nodes` | `array[string]`| Sequential camera nodes traversed | N/A | List of camera IDs | No | Member 2 | Inferred | `["CAM_S01_C001"]` |
| `spatial_semantics` | `string` | Coordinate frame designation | N/A | `"cityflow_world_planar_meters"` | No | AI City / Member 2 | Definition | `"cityflow_world_planar_meters"` |
| `temporal_semantics`| `string` | Temporal base designation | N/A | `"synchronized_video_seconds"` | No | AI City / Member 2 | Definition | `"synchronized_video_seconds"` |
| `segments` | `array[object]` | Pairwise transition segments between tracklets | N/A | List of segment records | No | Member 2 | Inferred | *See segment fields below* |

### Trajectory Segment Object (`segments[]`)

| Field Name | Type | Meaning & Semantics | Units | Allowed Values | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `segment_id` | `string` | Unique identifier for the transition | N/A | String format | No | Member 2 | Inferred | `"CAM_S01_C001_track_1_to_CAM_S01_C001_track_5"` |
| `time_difference_seconds` | `number` | Elapsed time between observation centers | seconds | Real number | Yes | Member 2 | Inferred | `14.25` |
| `status` | `string` | Association status | N/A | `"direct_connection"`, `"unassociated_camera"` | No | Member 2 | Inferred | `"direct_connection"` |
| `speed_mps` | `number` | Estimated transition speed | meters/sec | $\ge 0.0$ | Yes | Member 2 | Inferred | `null` |
| `plausibility` | `number` | Kinematic plausibility score | $[0.0, 1.0]$ | $0.0 \le p \le 1.0$ | Yes | Member 2 | Inferred | `1.0` |
| `start_observation.camera_id` | `string` | Camera of departure | N/A | Valid camera ID | No | Member 1 | Observed | `"CAM_S01_C001"` |
| `start_observation.timestamp` | `number` | Synchronized departure timestamp | seconds | $\ge 0.0$ | No | Member 1 / AI City | Observed / Sync | `2.125` |
| `start_observation.world_point` | `array[number]`| Ground-contact point $[X, Y]$ | meters | 2 floats or null | Yes | Member 2 / Homography | Inferred | `[-15.2, 42.1]` |
| `start_observation.image_point` | `array[number]`| Ground-contact bottom-center $[u, v]$ | pixels | 2 floats in frame | No | Member 1 | Observed | `[482.0, 715.0]` |

---

## 4. Camera State Contract (`camera_states.json`)

File: `UrbanTrack_Member2_Handoff/outputs/camera_states.json`  
Schema: `UrbanTrack_Member2_Handoff/schemas/camera_state_schema.json`  
Top-level Structure: Key-value map keyed by camera ID (`"CAM_S01_C001"`, `"CAM_S01_C002"`, `"CAM_S01_C003"`).

| Field Name | Type | Meaning & Semantics | Units | Allowed Values | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `camera_id` | `string` | Standardized camera identifier | N/A | Canonical camera ID | No | Member 1 | Observed | `"CAM_S01_C001"` |
| `camera_index` | `string` | Short camera index | N/A | `"c001"`, `"c002"`, `"c003"` | No | Member 1 | Observed | `"c001"` |
| `total_tracklets` | `integer` | Count of tracklets detected by Member 1 | count | $\ge 0$ | No | Member 1 | Observed | `107` |
| `observations_count`| `integer` | Total raw bounding boxes across tracklets | count | $\ge 0$ | No | Member 1 | Observed | `7229` (C001), `6702` (C002), `7879` (C003) |
| `video_duration_seconds` | `number` | Real video duration | seconds | $> 0.0$ | No | Member 1 | Observed | `195.5` (C001), `211.0` (C002), `199.6` (C003) |
| `fps` | `number` | Native video frame rate | frames/sec | $> 0.0$ | No | AI City | Observed | `10.0` |
| `reid_model` | `string` | Checkpoint used for feature extraction | N/A | Model identifier string | No | Member 1 | Provenance | `"osnet_x0_25_aicity"` |
| `reid_compatibility_status` | `string` | Compatibility guard status | N/A | `"compatible"`, `"incompatible_model_guarded"` | No | Member 2 | Inferred state | `"compatible"` (C001) / `"incompatible_model_guarded"` (C002) |
| `camera_reliability` | `number` | Calibrated empirical trust score | $[0.0, 1.0]$ | $0.0 \le r \le 1.0$ | No | Member 2 | Inferred prior | `0.7970` (C001), `0.7537` (C002), `0.6240` (C003) |
| `synchronization.offset_seconds` | `number` | Video start offset relative to sequence base | seconds | Real number | No | AI City Challenge | Synchronized | `0.0` (C001), `1.64` (C002), `2.049` (C003) |
| `synchronization.status` | `string` | Synchronization state | N/A | `"official_aicity_synchronized"`, `"unsynchronized_video_relative"` | No | Member 2 | Status | `"official_aicity_synchronized"` |
| `calibration.homography_available` | `boolean` | Presence of 3x3 homography matrix | N/A | `true`, `false` | No | Member 2 | Status | `true` |
| `calibration.reprojection_error_px` | `number` | Ground plane reprojection error | pixels | $\ge 0.0$ | Yes | AI City calibration | Calibration | `5.50` (C001), `11.44` (C002), `11.39` (C003) |
| `calibration.horizon_safeguard_active` | `boolean` | Active suppression of vanishing line singularities | N/A | `true`, `false` | No | Member 2 | Safeguard | `true` |
| `calibration.valid_projections_count` | `integer` | Count of tracklets with valid world $[X, Y]$ | count | $\ge 0$ | No | Member 2 | Inferred count | `95` (C001), `112` (C002), `152` (C003) |
| `calibration.horizon_rejected_count` | `integer` | Count of detections suppressed by horizon safeguard | count | $\ge 0$ | No | Member 2 | Inferred count | `12` (C001), `13` (C002), `0` (C003) |

---

## 5. Identity Graph Contract (`identity_graph.json`)

File: `UrbanTrack_Member2_Handoff/outputs/identity_graph.json`  
Schema: `UrbanTrack_Member2_Handoff/schemas/identity_graph_schema.json`  
Top-level Structure: Object with `nodes`, `edges`, and graph summary metrics.

### Node Object (`nodes[]`)
384 node objects representing Member 1 tracklets.

| Field Name | Type | Meaning & Semantics | Units | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `node_id` | `string` | Unique tracklet node identifier | N/A | No | Member 1 | Observed | `"CAM_S01_C001_track_1"` |
| `camera_id` | `string` | Camera where tracklet occurred | N/A | No | Member 1 | Observed | `"CAM_S01_C001"` |
| `local_track_id` | `integer` | Camera-local tracker identifier | integer ID | No | Member 1 | Observed | `1` |
| `start_time` | `number` | Synchronized start time | seconds | No | Member 1 / AI City | Observed / Sync | `0.0` |
| `end_time` | `number` | Synchronized end time | seconds | No | Member 1 / AI City | Observed / Sync | `4.25` |
| `assigned_identity_id` | `string` | Resolved global identity ID | N/A | No | Member 2 | Inferred | `"UT_ID_0002"` |
| `gt_vehicle_id` | `integer` | Ground truth ID (**EVALUATION ONLY**) | integer ID | Yes | AI City GT Linker | Evaluation-Only | `54` |

### Edge Object (`edges[]`)
36 confirmed pairwise association edges connecting tracklet nodes.

| Field Name | Type | Meaning & Semantics | Units | Nullable? | Source | Nature | Concrete Example |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `source` | `string` | Node ID of departure tracklet | N/A | No | Member 2 | Inferred | `"CAM_S01_C001_track_1"` |
| `target` | `string` | Node ID of arrival tracklet | N/A | No | Member 2 | Inferred | `"CAM_S01_C001_track_5"` |
| `weight` | `number` | Probabilistic evidence fusion association score | $[0.0, 1.0]$ | No | Member 2 | Inferred | `0.8523` |
| `decision` | `string` | Edge admission decision | N/A | No | Member 2 | Inferred | `"CONFIRMED"` |
| `reid_similarity` | `number` | Cosine similarity of OSNet vectors | $[-1.0, 1.0]$ | Yes | Member 2 | Inferred | `0.8523` |
| `temporal_feasible` | `boolean` | Temporal ordering validity | N/A | No | Member 2 | Inferred | `true` |
| `spatial_feasible` | `boolean` | Kinematic feasibility check | N/A | No | Member 2 | Inferred | `true` |
| `explanation` | `string` | Plain-English physical justification | N/A | No | Member 2 | Inferred | `"High visual appearance similarity (0.852); Spatiotemporally feasible"` |

---

## 6. Uncertainty & Calibration Contract (`uncertainty.json`)

File: `UrbanTrack_Member2_Handoff/outputs/uncertainty.json`  
Schema: `UrbanTrack_Member2_Handoff/schemas/uncertainty_schema.json`  

| Key / Field | Type | Meaning | Units | Value / State in Build |
| :--- | :--- | :--- | :--- | :--- |
| `calibration_model.method` | `string` | Calibration fitting algorithm | N/A | `"platt_scaling"` |
| `calibration_model.fitted_weights.A` | `number` | Logistic slope parameter | scalar | `0.5138` |
| `calibration_model.fitted_weights.B` | `number` | Logistic intercept parameter | scalar | `-3.7468` |
| `calibration_model.calibrated_brier` | `number` | Mean squared calibration error on holdout | $[0.0, 1.0]$ | `0.0227` (vs uncalibrated 0.1260) |
| `calibration_model.calibrated_ece` | `number` | Expected Calibration Error (ECE, 10 bins) | $[0.0, 1.0]$ | `0.0067` (vs uncalibrated 0.3127) |
| `calibration_model.zero_leakage_verified` | `boolean` | Verification of disjoint vehicle split | N/A | `true` (57 dev, 38 holdout) |
| `pairwise_decision_distribution.CONFIRMED` | `integer` | Admitted pairwise matches ($\ge 0.70$) | count | `36` |
| `pairwise_decision_distribution.AMBIGUOUS` | `integer` | Ambiguous candidates ($0.40 \le s < 0.70$) | count | `211` |
| `pairwise_decision_distribution.REJECTED` | `integer` | Rejected candidates ($< 0.40$ or incompatible) | count | `50,899` |

---

## 7. Anomaly Input Contract (`anomaly_input`)

Schema: `UrbanTrack_Member2_Handoff/schemas/anomaly_input_schema.json`  

Member 3's anomaly detection engine must consume diagnostic candidate records structured as follows:
- **`entity_id`** (`string`): Unique entity investigated (`"UT_ID_0002"`, `"CAM_S01_C002"`, etc.).
- **`entity_type`** (`string`): `"vehicle"`, `"camera"`, or `"segment"`.
- **`data_quality_status`** (`string`): `"valid"`, `"incomplete"`, or `"invalid"`. *Member 3 must never flag incomplete data as an operational anomaly.*
- **`reliability`** (`number` in $[0.0, 1.0]$): Sensor trust score.
- **`uncertainty`** (`number` in $[0.0, 1.0]$): Propagated inference uncertainty.
- **`signals`** (`array[object]`): List of physical telemetry excursions:
  - `signal_category` (`string`): `"speed_violation"`, `"reversal"`, `"teleportation"`, `"sensor_degradation"`.
  - `severity` (`string`): `"info"`, `"warning"`, `"critical"`.
  - `metric_name` (`string`): Name of measured diagnostic (e.g. `"inter_camera_speed_mps"`).
  - `measured_value` (`number`): Measured value.
  - `threshold_exceeded` (`number`): Theoretical boundary.
  - `explanation` (`string`): Physical explanation.

---

## 8. Summary of Normative Rules for Member 3

| Component | What Member 3 MUST Do | What Member 3 MUST NOT Do |
| :--- | :--- | :--- |
| **Identities** | Consume `identity_id` as global vehicle entity; group trips by `UT_ID_XXXX`. | Merge identities across cameras manually; assume all tracklets are merged. |
| **Ground Truth** | Restrict `gt_vehicle_id` exclusively to benchmark validation reports. | Expose `gt_vehicle_id` on user-facing dashboards or use it in business logic. |
| **Coordinates** | Plot $[X, Y]$ in local meter space or against CityFlow reference map. | Treat $[X, Y]$ as WGS84 GPS latitude/longitude or plot on Google Maps tiles. |
| **Timestamps** | Render elapsed playback time from $T_0 = 0.0$ seconds. | Convert seconds to arbitrary UTC calendar datetimes without explicit origin. |
| **Uncertainty** | Render unconfirmed singletons (`confidence: null`) as single-camera detections. | Overwrite `null` confidence with `1.0` or drop singletons from traffic counts. |
| **Anomalies** | Triage speed violations against camera synchronization and calibration limits. | Label physical speed excursions as guaranteed criminal/reckless driver events. |
