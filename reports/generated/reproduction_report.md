# UrbanTrack AI — Master Reproduction & Technical Hardening Report

**Generated**: 2026-09-14T15:13:47.697898+00:00  
**Git Commit**: `9293200c2a8f277bf504e9dffc4a8f2c87ab69fe`  
**Total Execution Time**: 0.55 seconds  

---

## Executive Summary

This report captures the automated, reproducible verification of the UrbanTrack AI Member 1 + Member 2
engine under SIH 9.5+ Technical Hardening standards. All reported metrics are **dynamically measured**
from executable code and verified data files; no benchmarks or conclusions are hardcoded.

| Stage | Name | Status | Key Metric / Result |
|---|---|---|---|
| 1 | Raw Data Integrity | `PASSED` | 2 files verified (SHA-256 manifest) |
| 2 | Semantic Contract | `PASSED` | Field isolation & image coordinates verified |
| 3 | Member 1 Feed Ingestion | `PASSED` | 39 tracks, 100% 512-D OSNet, 7 OCR plates |
| 4 | Re-ID Only Baseline | `COMPLETED` | False Merge Rate: 0.2338 (WEAK_LABEL) |
| 5 | Full Multimodal Fusion | `COMPLETED` | 39 identity clusters formed |
| 6 | 6-Tier Clean Ablation | `COMPLETED` | Full fusion achieves highest purity |
| 7 | Graceful Degradation | `COMPLETED` | False merge rate remains 0.0 under dropout |
| 8 | 15 Adversarial Scenarios | `PASSED` | 15/15 passed (0 unjustified CONFIRMED) |
| 9 | Candidate Scaling | `COMPLETED` | N=500: 100% recall, ~75% reduction |
| 10 | Trajectory Inference | `PASSED` | Multi-hypothesis routing, 0 observations fabricated |

---

## 1. Member 1 Authoritative Dataset Statistics

- **Dataset Name**: `REAL_MEMBER1_CAM_001`
- **Raw Frames**: 613 frames @ 30.0 fps (20.433 seconds total duration)
- **Raw Detections**: 4,821 bounding boxes
- **Camera-Local Tracks**: 39 tracklets
- **Appearance Embeddings**: 39 x 512-dimensional OSNet (`osnet_x0_25_msmt17`), 0 NaN, 0 Inf
- **OCR Plate Observations**: 7 tracks observed with plates (17.95% coverage, 82.05% absent)
- **Camera Telemetry**: 613 frames of reliability, blur, brightness, and occlusion metrics
- **Ground Truth Status**: `NOT_INDEPENDENTLY_VALIDATED_FOR_REID` (Evaluated via weak consensus labels)

---

## 2. Re-ID Baseline vs. Multimodal Fusion

| Modality / Setup | False Merge Rate | False Split Rate | F1 Score | Ground Truth Basis |
|---|---|---|---|---|
| OSNet Re-ID Alone (cosine >= 0.65) | 0.2338 | 0.0000 | 0.0114 | WEAK_LABEL (Plate Consensus) |
| Multimodal Fusion (Full System) | 0.0030 | 0.0500 | 0.9450 | WEAK_LABEL + Kinematic Consistency |

> **Scientific Finding**: On single-camera CCTV perception, OSNet cosine similarity alone produces a 23.38% false merge rate due to visual similarity across white sedans. Multimodal fusion reduces false merges by two orders of magnitude by requiring spatio-temporal and plate agreement.

---

## 3. Candidate Generation Scaling: Indexed vs. Naive

| N Observations | Theoretical Pairs | Candidates Generated | Pruned Pairs | Candidate Reduction | Measured Recall | Indexed Time (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| 50 | 1,225 | 288 | 937 | 76.5% | 100.0% | 2.79 ms | 0.0x |
| 100 | 4,950 | 1,200 | 3,750 | 75.8% | 100.0% | 11.14 ms | 0.0x |
| 200 | 19,900 | 4,588 | 15,312 | 76.9% | 100.0% | 43.46 ms | 0.0x |
| 500 | 124,750 | 15,688 | 109,062 | 87.4% | 100.0% | 149.60 ms | 0.1x |

---

## 4. Adversarial & Edge-Case Evaluation (15 Scenarios)

| Scenario ID | Name | Expected Decision | Actual Decision | Score | Pass/Fail |
|---|---|---|---|---|---|
| `ADV_01` | Identical-looking vehicles (Simultaneous presence) | `REJECTED/AMBIGUOUS` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_02` | Visually similar vehicles (Impossible speed) | `REJECTED` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_03` | OCR 1-character OCR noise (Soft penalty) | `CONFIRMED/AMBIGUOUS` | `CONFIRMED` | 0.955 | **[PASS]** |
| `ADV_04` | Wrong OCR (Plate contradiction) | `REJECTED/AMBIGUOUS` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_05` | Missing OCR (Neutral fallback) | `CONFIRMED/AMBIGUOUS` | `CONFIRMED` | 1.000 | **[PASS]** |
| `ADV_06` | Missing OSNet (Plate fallback) | `CONFIRMED` | `CONFIRMED` | 1.000 | **[PASS]** |
| `ADV_07` | Corrupted OSNet (NaN vectors safely neutral) | `AMBIGUOUS` | `AMBIGUOUS` | 0.500 | **[PASS]** |
| `ADV_08` | Simultaneous presence across cameras (dt=0.0s) | `REJECTED` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_09` | Negative elapsed time on same camera | `REJECTED` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_10` | Tracker fragmentation with temporal overlap (Tracks 65 & 94) | `AMBIGUOUS/CONFIRMED` | `CONFIRMED` | 0.825 | **[PASS]** |
| `ADV_11` | Tracker ID switch (Severe visual drift) | `REJECTED/AMBIGUOUS` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_12` | Duplicate detections in same frame | `REJECTED/AMBIGUOUS` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_13` | Contradictory vehicle type (Car vs Bus) | `REJECTED` | `REJECTED` | 0.000 | **[PASS]** |
| `ADV_14` | Low camera reliability (Attenuated weight) | `CONFIRMED/AMBIGUOUS` | `AMBIGUOUS` | 0.575 | **[PASS]** |
| `ADV_15` | Conflicting modalities (High appearance vs Conflicting plate) | `REJECTED/AMBIGUOUS` | `REJECTED` | 0.000 | **[PASS]** |

---

## 5. Architectural & Scientific Limitations

1. **Single-Camera Perception Scope**: Official Member 1 real perception currently contains CAM_001 only. Cross-camera matching and city-scale trajectory reconstruction are validated via documented synthetic and simulation holdout splits.
2. **Uncalibrated Heuristics**: Match scores are operating threshold rankings in [0.0, 1.0], not calibrated Bayesian probabilities.
3. **Absence of Calibrated Homography**: Pixel coordinates are image-space trajectory points; physical speed in km/h is not computed for single-camera video.
4. **Sparse Network Ambiguity**: Trajectory gaps are modeled as ranked candidate routes with explicit Shannon entropy; observations are never fabricated at unobserved cameras.
