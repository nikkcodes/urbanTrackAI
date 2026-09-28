# UrbanTrack AI — Member 2 Official Handoff to Member 3

**Layer**: Member 2 — Probabilistic Multi-Camera Vehicle Intelligence Engine  
**Target Consumer**: Member 3 — City Mobility Analytics, Simulation, GIS & Dashboard Layer  
**Validation Base**: AI City Challenge 2022 Track 1 / CityFlowV2 Real Perception Slice (`CAM_S01_C001`, `CAM_S01_C002`, `CAM_S01_C003`)  
**Package Path**: `UrbanTrack_Member2_Handoff/`  
**Manifest**: `HANDOFF_MANIFEST.json`

---

## 1. Executive Summary

This handoff package provides the official, verified interface between **Member 2 (Probabilistic Vehicle Intelligence Engine)** and **Member 3 (Downstream City Analytics, Traffic Simulation & Dashboard)**.

Member 2 has completed the ingestion, candidate pruning, multimodal probabilistic evidence fusion, identity graph clustering, world-space trajectory reconstruction, and holdout calibration on the 3-camera AI City Challenge 2022 dataset.

Member 3 should consume the structured outputs, schemas, and calibration ledgers in this package to power:
- City origin-destination (OD) flow estimation
- Traffic bottleneck and queue analysis
- GIS map visualization and trajectory playback
- Anomaly investigation and diagnostic reporting
- Counterfactual scenario simulation

Member 3 **MUST NOT** re-implement perception, Re-ID extraction, pairwise fusion, or identity graph clustering.

---

## 2. Pipeline Boundary

```
┌──────────────────────────────────────────────────────────────────┐
│                   MEMBER 1: PERCEPTION LAYER                     │
│  - YOLOv8 frame-level bounding boxes (21,810 observations)       │
│  - DeepSORT camera-local tracklets (384 tracklets)               │
│  - OSNet appearance embeddings (322 512-D unit-norm vectors)     │
│  - Camera telemetry (blur, brightness, occlusion, detection conf)│
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (Canonical Observation Feed)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                 MEMBER 2: INTELLIGENCE ENGINE                    │
│  - Official AI City timestamp synchronization (+0.0s, +1.64s, +2.049s)│
│  - Homography ground projection (359/384 tracklets with valid world) │
│  - Vanishing horizon safeguard (|W| < 0.50 runaway suppression)  │
│  - Prior camera reliability estimation                           │
│  - O(N log N) spatiotemporal candidate pruning (30.45% pruned)   │
│  - Multimodal probabilistic evidence fusion (C002 Re-ID guard)   │
│  - Identity graph clustering (384 nodes, 36 edges, 355 clusters) │
│  - World-space trajectory inference (image/world dual mode)      │
│  - Platt scaling calibration on disjoint vehicle holdout         │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ (UrbanTrack_Member2_Handoff/)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                 MEMBER 3: DOWNSTREAM APPLICATIONS                │
│  - City Mobility Analytics & OD Matrix Generation                │
│  - Traffic Bottleneck & Congestion Diagnostics                   │
│  - Spatial GIS Visualization & Fleet Playback                    │
│  - Anomaly & Incident Investigation Engine                       │
│  - Counterfactual Network Simulation & Digital Twin              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. What Member 2 Does

1. **Camera Reliability Modeling**: Evaluates empirical sensor trustworthiness based on detection confidence stability, optical clarity (blur/brightness), and metadata completeness.
2. **Spatiotemporal Candidate Pruning**: Filters the 73,536 theoretical all-pairs search space down to 51,146 plausible candidate pairs in 9.19 ms using physical speed horizons and vehicle type consistency.
3. **Cross-Model Re-ID Protection**: Audits embedding provenance. Automatically disables visual similarity between incompatible latent spaces (`osnet_x0_25_msmt17` on C002 vs `osnet_x0_25_aicity` on C001/C003), preventing catastrophic false identity merges.
4. **Horizon-Safe World Projection**: Transforms image bounding-box ground-contact points into metric planar world coordinates (`cityflow_world`) via official 3x3 homographies, with strict suppression of vanishing-line mathematical singularities ($|W| < 0.50$). 359 of 384 tracklet observations yielded valid world coordinates; 25 were horizon-suppressed.
5. **Global Identity Graph Assembly**: Formulates connected identity hypotheses using deterministic graph components, with transitive contradiction resolution preventing inconsistent vehicle types or impossible travel speeds from merging.
6. **Probabilistic Calibration**: Scales heuristic fusion scores into calibrated match probabilities using Platt logistic regression fitted on disjoint vehicle DEV/HOLDOUT splits (zero vehicle leakage; Brier: 0.0227, ECE: 0.0067).
7. **Trajectory Reconstruction**: Formulates chronological trajectory waypoints, transition velocity estimates, and gap intervals.

---

## 4. What Member 2 Does NOT Do

Member 2 **DOES NOT** own or implement:
- **Frontend / UI**: Dashboard layouts, React/Vue components, web servers, or WebSocket streams.
- **GIS Cartography**: Map tile rendering, Mapbox/Leaflet overlays, or GPS georeferencing.
- **Macroscopic Traffic Models**: BPR volume-delay functions, cell transmission models, or network-wide equilibrium assignment.
- **Counterfactual Simulation Engine**: Simulating road closures, lane reductions, signal retiming, or diverted traffic flow.
- **Anomaly Investigation UI**: End-user alert triage workflows or investigation ticketing.

All of the above are the exclusive responsibility of **Member 3**.

---

## 5. What Member 3 Can Trust

Member 3 can safely rely on the following verified outputs:
1. **`outputs/inferred_identities.json`**: 355 inferred vehicle identity hypotheses with explicit admission status (`confirmed` vs `unconfirmed_singleton`).
2. **`outputs/camera_states.json`**: Real operational telemetry, prior reliability scores, synchronization offsets, and homography calibration status for C001, C002, and C003.
3. **`outputs/trajectories.json`**: Chronologically ordered observation sequences and kinematic transition measurements.
4. **`outputs/identity_graph.json`**: Pairwise association edges with complete evidence ledgers and plain-English explanations.
5. **`outputs/uncertainty.json`**: Platt calibration parameters ($A=0.5138, B=-3.7468$) and verified decision distribution.
6. **Zero Ground-Truth Leakage**: Pre-inference audit passed; zero evaluation labels were exposed to inference.
7. **Conservative False Merge Rate**: Across 28,380 cross-camera negative pairs, False Merge Rate is strictly **0.000000**.

---

## 6. What Member 3 Must Treat as Uncertain

Member 3 **MUST NOT** collapse probabilistic uncertainty into binary ground truth:
1. **Ambiguous Identity Matches (`AMBIGUOUS`)**: Candidate pairs whose score sits between 0.40 and 0.70 represent incomplete evidence. Member 3 must display these as alternative hypotheses, not confirmed matches.
2. **Unconfirmed Singletons (`unconfirmed_singleton`)**: 337 out of 355 identity hypotheses consist of single-camera tracklets. Their confidence is explicitly `null` (not 1.0).
3. **Unassociated Trajectory Cameras**: In the AI City Challenge dataset, camera locations are not georeferenced to an OpenStreetMap road network. Trajectory segments carry status `"unassociated_camera"` and `total_distance_m: 0.0`.
4. **Near-Horizon Coordinates**: Observations near camera horizon vanishing lines are marked `"World position uncertain"` (`world_x=None, world_y=None`).
5. **Diagnostic Speed Excursions**: Measured speeds exceeding 120 km/h (occurring in 26.9% of world-coordinate transitions due to homography ground-plane projection noise) are **data quality artifacts**, NOT verified driver speeding violations.
6. **Unavailable Modalities**: License plate text is privacy-censored in the benchmark; appearance evidence is unavailable on C002 transitions.

---

## 7. What Member 3 Must NOT Claim

To preserve absolute scientific and technical integrity, Member 3 **MUST NEVER CLAIM**:
1. **DO NOT CLAIM 46-camera validation**: The empirical MVP is strictly validated on **3 cameras** (`C001`, `C002`, `C003`). The architecture is extensible to 46 cameras, but has not been validated on 46 cameras.
2. **DO NOT CLAIM readable license plate accuracy**: Plates in the AI City Challenge benchmark are privacy-censored. No OCR accuracy can be claimed.
3. **DO NOT CLAIM WGS84 GPS mapping**: Coordinates are benchmark planar metric coordinates (`cityflow_world`), not GPS latitude/longitude.
4. **DO NOT CLAIM proven routes where only hypotheses exist**: Without an active OSM road graph, trajectory segments connect camera sighting endpoints, not microscopic street turns.
5. **DO NOT CLAIM GT IDs as production identities**: AI City vehicle IDs (e.g. `54`) are evaluation-only ground-truth labels. Member 2's inferred identities are `UT_ID_XXXX`.
6. **DO NOT CLAIM production traffic predictions from synthetic simulations**: All simulation outputs generated by Member 3 must be explicitly tagged as `[SIMULATED]`.

---

## 8. Package Directory Guide

| Directory / File | Description |
| :--- | :--- |
| `HANDOFF_MANIFEST.json` | Catalog of all files, hashes, data scopes, and version metadata. |
| `INTERFACE_CONTRACT.md` | Normative field-by-field specification of all JSON structures. |
| `DATA_DICTIONARY.md` | Detailed semantic dictionary answering the 8 canonical field questions. |
| `SOURCE_AND_PROVENANCE.md` | Lineage tracking for Member 1, AI City, Member 2, and GT data. |
| `VALIDATION_STATUS.md` | Empirical test results, ablations, calibration, and benchmarks. |
| `RUN_INSTRUCTIONS.md` | Exact commands to inspect, validate, and test the handoff. |
| `schemas/` | Formal JSON Schema (Draft-07) specifications for all interfaces. |
| `outputs/` | Real Member 2 inference outputs ready for ingestion. |
| `examples/` | Real confirmed, ambiguous, rejected, and singleton examples. |
| `validation/` | Exact copies of all final validation and ablation artifacts. |
