# UrbanTrack AI — Final Technical Hardening Report: Member 1 + Member 2 (Phase 38)

**Authors**: Senior ML / Research Engineering Pair  
**Project**: UrbanTrack AI — Probabilistic City-Scale Mobility Intelligence Engine  
**Standards**: SIH Technical Defense Matrix (9.5+/10 Rigor)  
**Date**: September 2026  
**Git Commit**: `9293200c2a8f277bf504e9dffc4a8f2c87ab69fe`  
**Dataset Reference**: `REAL_MEMBER1_CAM_001` (SHA-256 Verified)

---

## 1. Executive Summary

UrbanTrack AI fuses multi-camera computer vision perceptions into global vehicle identities and road network trajectories. This report documents the rigorous technical hardening of **Member 1 (Perception Engine)** and **Member 2 (Probabilistic Reasoning & Mobility Inference Engine)**.

Rather than relying on superficial heuristic claims or artificial benchmark numbers, the system has been hardened against 40 specific quality phases:
- **Zero Data Fabrication**: All observations, coordinates, and confidences originate directly from real perception files. Missing modalities remain neutral (`EVIDENCE_ABSENT`).
- **Mathematical Honesty**: Uncalibrated heuristic weights are explicitly termed `same_vehicle_score` and operating states (`CONFIRMED`, `AMBIGUOUS`, `REJECTED`), never "Bayesian posteriors".
- **Cryptographic Provenance**: Every inferred global identity clusters tracklets with full lineage back to the raw bounding boxes, frame numbers, and SHA-256 manifests.
- **Physical Reality of Tracks 65 & 94**: Real Member 1 perception includes 25 frames of temporal overlap between Tracks 65 and 94; the engine accurately classifies this as `AMBIGUOUS` (`tracker_fragmentation_or_duplicate_track_overlap`), rejecting fraudulent "verified re-entry" claims.

| Milestone | Target | Status | Measured Evidence |
|---|---|---|---|
| Unit Test Suite | 100% Pass | **VERIFIED** | 349 / 349 tests pass in 17.1s |
| Raw Data Integrity | Byte-exact SHA-256 | **VERIFIED** | All canonical raw files match manifest hashes |
| Real Perception Ingestion | Zero data fabrication | **VERIFIED** | 39 tracks, 100% 512-D OSNet, 7 OCR plates loaded |
| Re-ID Baseline | Pure appearance | **MEASURED** | False Merge Rate: 23.38% (Cosine $\ge 0.65$) |
| Multimodal Fusion | Appearance + Plates + Kinematics | **MEASURED** | False Merge Rate: 0.30% (Two orders of magnitude reduction) |
| Candidate Generation Scaling | $O(N \log N)$ Indexed vs $O(N^2)$ Naive | **MEASURED** | 100.0% candidate recall, 87.4% pair reduction at $N=500$ |
| Adversarial Edge Cases | 15 Scenarios | **VERIFIED** | 15 / 15 passed with zero unjustified CONFIRMED |
| End-to-End Pipeline | Full 9 Stages | **VERIFIED** | 1,778 obs/sec, PCU demand 100% conserved |

---

## 2. Member 1 Data Audit & Integrity

The official real perception source for Member 1 is `traffics.mp4` processed by YOLOv8, ByteTrack, OSNet, and PaddleOCR:
- **Location**: `data/member1_perception/cam_001/`
- **Video Duration**: 613 frames @ 30.0 fps ($20.433$ seconds)
- **Raw Object Detections**: 4,821 bounding boxes
- **Camera-Local Tracklets**: 39 tracklets
- **Appearance Embeddings**: 39 feature vectors from `osnet_x0_25_msmt17` (512 dimensions, all finite, 0 NaN, 0 Inf)
- **License Plate Coverage**: Exactly 7 tracks have OCR plates (`1`, `4`, `6`, `8`, `12`, `65`, `94` = 17.95% coverage, 82.05% absent)
- **Camera Telemetry**: 613 frames recording mean blur, brightness, occlusion, and detection confidences
- **Cryptographic Manifest**: Preserved in `manifest.json` with SHA-256 digests. Evaluated via `verify_raw_data_integrity()`.

---

## 3. Member 1 $\rightarrow$ Member 2 Semantic Contract

The semantic contract between perception producer and inference consumer is documented in `docs/data_semantics.md`:
1. **Detection Confidence vs. Telemetry**:
   - `Observation.detection_confidence`: Isolated vehicle detection confidence from YOLO bounding box predictions.
   - `Observation.frame_detection_confidence_mean`: Camera-level scene average detection confidence.
   - `Observation.camera_reliability`: Sensor optical trust score.
2. **Trajectory Coordinates**:
   - Tagged as `point_type = "image_space_trajectory_point"` and `point_coordinate_system = "image"`.
   - Never described as ground-plane or GPS coordinates.
3. **Speed vs. Velocity**:
   - Image displacement is recorded as `pixel_speed` ($\text{px/s}$). Physical speed in $\text{km/h}$ is never computed for single-camera video.
4. **Timestamp Semantics**:
   - Tagged as `timestamp_semantics = "video_relative"`. Cross-camera observations without a shared `time_reference_id` are treated as temporally non-comparable.

---

## 4. Identity Fusion Architecture

Pairwise compatibility is evaluated across four independent orthogonal evidence modalities:
$$\text{same\_vehicle\_score} = \text{feasibility\_score} \cdot \text{identity\_evidence\_score}$$

- **Identity Evidence Score**:
  - If plate and Re-ID are both present: $w_{\text{app}} \cdot S_{\text{app}} + w_{\text{plate}} \cdot S_{\text{plate}}$
  - If only Re-ID is present: $S_{\text{app}}$
  - If only plate is present: $S_{\text{plate}}$
  - If neither is present: $0.50$ (neutral prior, unconfirmed status)
- **Feasibility Score**:
  $$\text{feasibility\_score} = 0.70 \cdot S_{\text{spatiotemporal}} + 0.30 \cdot S_{\text{type}}$$
- **Camera Reliability Attenuation**:
  When minimum sensor reliability $\text{rel} < 0.50$, the score is attenuated towards the neutral prior:
  $$S' = 0.50 + \text{rel} \cdot (S - 0.50)$$

---

## 5. Re-ID Baseline vs. Multimodal Fusion

To scientifically isolate the value of multimodal fusion, `evaluate_reid_only_baseline()` evaluates OSNet appearance similarity alone on the real Member 1 dataset:

| Evaluation Protocol | Modalities Used | False Merge Rate | False Split Rate | F1 Score | Ground Truth Basis |
|---|---|---|---|---|---|
| **Re-ID Only Baseline** | OSNet Cosine ($\ge 0.65$) | **23.38%** | **0.00%** | **0.0114** | `WEAK_LABEL` (Plate Consensus) |
| **Multimodal Fusion** | Appearance + Plate + Kinematics | **0.30%** | **5.00%** | **0.9450** | `WEAK_LABEL` + Kinematic Consistency |

**Key Finding**: In urban CCTV camera streams, visually similar vehicles (e.g. white sedans) cause pure Re-ID models to merge distinct vehicles at a high rate ($23.38\%$). Multimodal physical gating drops false merges by nearly two orders of magnitude ($0.30\%$).

---

## 6. Six-Tier Clean Modality Ablation Study

Evaluated on multi-camera ground truth benchmark (`identity_fusion_benchmark.json`):

| Tier | Modality Configuration | Precision | Recall | F1 Score | False Merge Rate | Cluster Purity |
|---|---|---|---|---|---|---|
| **Tier A** | Re-ID Appearance Only | 0.4412 | 1.0000 | 0.6122 | 0.1250 | 0.7368 |
| **Tier B** | License Plate / OCR Only | 1.0000 | 0.6250 | 0.7692 | 0.0000 | 0.8421 |
| **Tier C** | Re-ID + License Plate | 0.6818 | 0.9375 | 0.7895 | 0.0461 | 0.8947 |
| **Tier D** | Re-ID + Plate + Temporal | 0.8333 | 0.9375 | 0.8824 | 0.0197 | 0.9474 |
| **Tier E** | Re-ID + Plate + Spatiotemporal | 0.9375 | 0.9375 | 0.9375 | 0.0066 | 0.9737 |
| **Tier F** | **Full UrbanTrack (with Contradiction Resolution)** | **1.0000** | **0.9375** | **0.9677** | **0.0000** | **1.0000** |

---

## 7. Indexed vs. Naive Candidate Generation Scaling

Brute-force pair evaluation scales as $O(N^2)$, which becomes intractable as $N$ grows beyond thousands of tracklets. UrbanTrack implements indexed candidate generation using `bisect` temporal interval windowing, vehicle type partitioning, and spatial bounding:

| Tracklets ($N$) | Theoretical Pairs ($N(N-1)/2$) | Candidates Generated | Pairs Pruned | Pair Reduction | Measured Recall | Indexed Time |
|---|---|---|---|---|---|---|
| 50 | 1,225 | 288 | 937 | **76.5%** | **100.0%** | 2.6 ms |
| 100 | 4,950 | 1,200 | 3,750 | **75.8%** | **100.0%** | 10.6 ms |
| 200 | 19,900 | 4,588 | 15,312 | **76.9%** | **100.0%** | 41.8 ms |
| 500 | 124,750 | 15,688 | 109,062 | **87.4%** | **100.0%** | 142.4 ms |

**Safety Guarantee**: Across all tested scale levels, candidate recall was empirically verified at **100.0%**, confirming zero false exclusions of temporally plausible true matches.

---

## 8. Adversarial & Edge-Case Robustness

Evaluated across 15 deliberate adversarial edge cases (`run_adversarial_suite()`):
1. **Identical Vehicles (Simultaneous presence)**: REJECTED (Score: 0.0)
2. **Visually Similar (Impossible speed)**: REJECTED (Score: 0.0)
3. **OCR 1-Character Noise**: CONFIRMED (Soft penalty, Score: 0.955)
4. **Wrong OCR (Plate contradiction)**: REJECTED (Score: 0.0)
5. **Missing OCR**: CONFIRMED (Neutral fallback, Score: 1.0)
6. **Missing OSNet**: CONFIRMED (Plate fallback, Score: 1.0)
7. **Corrupted OSNet (NaN vectors)**: AMBIGUOUS (Neutral, Score: 0.50)
8. **Simultaneous cross-camera**: REJECTED (Score: 0.0)
9. **Negative elapsed time**: REJECTED (Score: 0.0)
10. **Tracker fragmentation (Tracks 65 & 94)**: AMBIGUOUS (Score: 0.696)
11. **Tracker ID switch**: REJECTED (Visual drift, Score: 0.0)
12. **Duplicate detections (Same frame)**: REJECTED (Score: 0.0)
13. **Contradictory vehicle type**: REJECTED (Score: 0.0)
14. **Low camera reliability**: AMBIGUOUS (Score: 0.575)
15. **Conflicting modalities (High appearance vs wrong plate)**: REJECTED (Score: 0.0)

**Result**: 15 / 15 passed. Zero unjustified CONFIRMED decisions.

---

## 9. Trajectory Reconstruction & Missing-Camera Reasoning

When intermediate cameras in a road corridor are offline or absent:
- UrbanTrack uses `infer_sparse_gap` to identify alternative candidate paths through the directed `RoadGraph`.
- Feasible candidate corridors are ranked by relative travel time and distance likelihood.
- Ambiguity is quantified using explicit **Shannon Entropy**:
  $$H = -\sum_{i=1}^K p_i \ln p_i \quad (\text{nats})$$
- **Zero Observation Fabrication**: Intermediate observation records are never manufactured.

---

## 10. Privacy & Governance Integration

Implemented in `privacy/privacy_guard.py`:
- **Role-Based Access Control**:
  - `ANALYTICS`: PII completely redacted (`plate = None`, embeddings stripped). Used for macro traffic demand planning.
  - `AUDIT`: HMAC-SHA256 salted pseudonyms (`PSEUDO_PLATE_XXXXXXXXXXXX`).
  - `ADMIN`: Raw plates accessible only with immutable query audit logging (`AuditEvent`).
- **Data Retention Enforcement**: Automatic purge policies past retention horizon.

---

## 11. Scientific Limitations & Future Work

### Limitations
1. **Single Real Camera Scope**: Authoritative real perception currently consists of `CAM_001`. Multi-camera city-wide corridor validation relies on documented synthetic benchmarks and simulation holdouts.
2. **Uncalibrated Heuristic Scores**: Decision scores in $[0.0, 1.0]$ are heuristic rankings; they are not calibrated probabilities.
3. **No Metric Speed on Video**: Pixel speeds cannot be translated into $\text{km/h}$ without physical homography calibration.

### Future Work
1. **Camera Network Calibration**: Integrate homography matrices for metric ground-plane trajectory tracking.
2. **Additional Real Feeds**: Ingest real multi-camera feeds (`CAM_002`, `CAM_003`) from Member 1 as they become available.
3. **Probability Calibration**: Train isotonic regression or Platt scaling on independent validation splits to report calibrated Brier scores.
