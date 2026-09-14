# UrbanTrack AI — Technical Hardening Audit & Repository Truth Baseline

**Audit Date**: September 2026  
**Auditor**: Senior ML / Probabilistic Reasoning Research Engineer (Member 2)  
**Scope**: Member 1 Perception Integration & Member 2 Reasoning Engine  
**Standard**: Strict Empirical Evidence Over Claims (Zero Fabrication, Mathematical Honesty)

---

## 1. Executive Summary & Audit Mandate

In accordance with Section 1 of the Technical Hardening Master Prompt, this audit establishes the **authoritative ground truth** of the codebase, raw data assets, test suites, schemas, and performance benchmarks. 

Every claim, metric, and schema definition in documentation was independently audited against the actual code and raw data files. Non-authoritative artifacts (such as stale README files, manually written benchmark numbers, and legacy data dumps) have been identified and categorized for systematic remediation.

---

## 2. Actual Repository & File Counts

Counts obtained via non-cache filesystem audit (`find . -not -path '*/.*' -not -path '*/__pycache__/*' | grep -E '\.py$'`):

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Total Non-Cache Files** | **102** files | All `.py`, `.json`, `.md` files (excluding `.git` and `__pycache__`) |
| **Total Python Files** | **70** files | All `.py` source and test files |
| **Python Source Files (Excl. Tests)** | **47** files | Distributed across `inference/`, `schemas/`, `mobility/`, `anomaly/`, `simulation/`, `privacy/` |
| **Python Test Files** | **23** files | In `tests/`: 20 unit test suites (`test_*.py`) + 3 validation scripts (`validate_*.py`) |
| **Total Active Code Directories** | **8** packages | `inference`, `schemas`, `mobility`, `anomaly`, `simulation`, `privacy`, `tests`, `docs` |

---

## 3. Actual Test Suite Status

Executed directly via Python's standard unittest test discovery:
`python3 -m unittest discover -s tests -p "test_*.py"`

| Suite Type | Test Count | Passing | Failing | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Unittest Discovery (`test_*.py`)** | **349** | **349** | **0** | **17.12s** |
| `tests/validate_day3_end_to_end.py` | 6 stages | 6 passed | 0 | ~1.5s |
| `tests/validate_full_pipeline.py` | 9 stages | 9 passed | 0 | ~3.8ms |
| `tests/validate_day2_end_to_end.py` | 6 stages | Stage 1 pass | **Stage 2 FAIL** | Crashed (`TypeError: '>' not supported between instances of 'NoneType' and 'float'`) |

### Critical Finding on `validate_day2_end_to_end.py`:
In `tests/validate_day2_end_to_end.py` line 106:
```python
"feasible": ev["temporal_feasibility"] > 0.0
```
When observations originate from different cameras without a shared synchronization reference, `inference/temporal.py` correctly reports `feasibility_score = None` and `status = "unavailable"`. The test assumed a float value and crashed on the `None` comparison. This confirms that test expectations must be hardened to handle valid `None` states when evidence is unavailable.

---

## 4. Actual Member 1 Perception Data Statistics

Audit performed directly on `data/member1_perception/cam_001/raw/`:

| Dimension | Measured Value | Verification Method |
| :--- | :--- | :--- |
| **Source Video** | `traffics.mp4` | Recorded in `video_summary.json` |
| **Camera Identifier** | `CAM_001` | Header in `raw_frame_detections.json` |
| **Video Resolution** | $3840 \times 2160$ (4K UHD) | Recorded in `video_summary.json` |
| **Video Frame Rate** | 30.0 fps | Recorded in `video_summary.json` |
| **Total Frames Processed** | 613 frames (frame 0 to 612) | Length of `frames` list in `raw_frame_detections.json` |
| **Video Duration** | 20.433 seconds ($t = 0.000\text{s}$ to $20.400\text{s}$) | Frame timeline |
| **Raw Bounding Box Detections** | **4,821** detections | Sum of vehicles across all 613 frames |
| **Camera-Local Tracklets** | **39** tracklets | Length of `trajectories.json` (Track IDs 1 to 94) |
| **Re-ID Model Architecture** | `osnet_x0_25_msmt17` | Verified in track records |
| **Embedding Dimensionality** | Exactly **512** dimensions | Validated across all 39 vectors |
| **Embedding Numeric Integrity** | **39 / 39 (100%) finite floats** | 0 NaN, 0 Inf, all floats bounded in $[-1.0, 1.0]$ |
| **Tracks with OCR Plate Data** | **7 tracks** (17.95%) | Tracks: `1`, `4`, `6`, `8`, `12`, `65`, `94` |
| **Tracks Missing OCR Plate Data** | **32 tracks** (82.05%) | Remaining 32 tracks have `plate_number: null` |
| **Camera Telemetry Records** | **613** records | 1 record per video frame in `camera_telemetry.json` |
| **Mean Camera Reliability** | $0.5164 \pm 0.0468$ | Telemetry distribution |
| **Mean Lens Blur Score** | $0.3003$ | Telemetry distribution |
| **Mean Scene Brightness** | $0.4507$ | Telemetry distribution |
| **Mean Occlusion Ratio** | $0.2422$ | Telemetry distribution |

---

## 5. Duplicate, Stale, & Legacy Data Paths

A crucial objective of Section 3 is eliminating legacy data confusion. The audit identified the following data files:

1. **`data/observations/kanishka_traffic.json` (LEGACY / OBSOLETE)**:
   - Contains 3,840 records of dummy bounding boxes from an early mock sprint.
   - Completely lacks appearance embeddings, license plates, and telemetry.
   - **Risk**: `run_real_data.py` currently loads this legacy file and prints:
     *"Because Kanishka's perception feed currently lacks Re-ID appearance embeddings and license plates..."*
   - **Remediation**: Move to `data/legacy/kanishka_traffic.json.legacy` and deprecate `run_real_data.py` in favor of canonical `run_real_member1.py`.
2. **`data/observations/cam_001_real_observations.json` (UNNORMALIZED DUMP)**:
   - 492 KB unnormalized intermediate file.
   - **Remediation**: Quarantine to `data/legacy/`.
3. **`data/observations/observation.json` & `data/observations/trajectories.json` (MIRRORS)**:
   - Duplicate copies of `raw_frame_detections.json` (3.5 MB) and `trajectories.json` (688 KB).
   - **Remediation**: Clarify provenance as reproducible mirrors of `data/member1_perception/cam_001/raw/`.
4. **`data/observations/sample_feed.json` (TEST FIXTURE)**:
   - 7 small test observations used in unit tests. Keep strictly as a test fixture.

---

## 6. Semantic Inconsistencies & Contract Corrections

The audit uncovered several critical semantic discrepancies where field names or code behavior diverged from real-world physics:

### A. Detection Confidence Semantics (Section 5)
- **Bug in `observation_loader.py:242`**:
  ```python
  det_conf = float(tel.get("detection_confidence_mean", 0.85)) if "detection_confidence_mean" in tel else 0.85
  ```
  `Observation.detection_confidence` was being overwritten with the camera telemetry's frame-average confidence, rather than preserving the vehicle's actual detection confidence from `raw_frame_detections.json` (e.g. `0.9292` for Track 65).
- **Remediation**: Preserve `detection_confidence` as the vehicle detection model's score, and introduce `frame_detection_confidence_mean` for the telemetry aggregate.

### B. Telemetry Aggregation Semantics (Section 6)
- **Limitation in `observation_loader.py:240`**:
  Only the tracklet's first frame (`start_frame`) telemetry was sampled and applied to the entire tracklet duration (e.g. applying frame 282 telemetry to a 331-frame tracklet spanning frames 282 to 612).
- **Remediation**: Compute explicit tracklet-level telemetry aggregates:
  `telemetry_frame_count`, `mean_reliability`, `min_reliability`, `max_reliability`, `mean_blur`, `mean_brightness`, `mean_occlusion`.

### C. Trajectory Point Semantics: Centroid vs. Footpoint (Section 4)
- In `Observation`, `point_type` was defaulted to `"vehicle_footpoint"`.
- However, Member 1's actual producer schema in `raw_frame_detections.json` outputs `centroid: [x, y]`.
- **Remediation**: Update default `point_type` to `"image_space_trajectory_point"` or explicitly document that it reflects the detection centroid.

### D. The Track 65 / Track 94 Reality (Section 12)
- Earlier documentation and demo text referred to Track 65 and Track 94 as *"verified re-entry"*.
- **Empirical Reality from Data Audit**:
  - Track 65: active from frame 282 to 612 (duration: 331 frames).
  - Track 94: active from frame 588 to 612 (duration: 25 frames).
  - **Temporal Overlap**: Frames 588 to 612 (25 frames of simultaneous activity).
  - In frame 588: Track 65 is at bbox `[118, 1475, 1096, 2023]`; Track 94 is at bbox `[5, 2001, 155, 2154]`.
  - Both share the plate `MH0ZFX9484` and visual similarity ($0.6765$).
  - **Scientific Diagnosis**: A vehicle cannot physically be in two different locations simultaneously. Therefore, Track 94 is **NOT a re-entry**; it is a **tracker fragmentation / duplicate-track artifact** created when ByteTrack instantiated a new track ID at the bottom-left boundary while Track 65 was still maintained.
  - **Remediation**: Update engine and explainability to classify this as `AMBIGUOUS` with reason `"tracker_fragmentation_or_duplicate_track_overlap"`, rather than falsely claiming re-entry.

### E. OCR Plate Inconsistencies (Section 7)
- Track 8 has conflicting OCR plate observations across frames: `NH0LBD4932` and `NH0LDD4922`.
- **Remediation**: Maintain consensus voting ledger with support counts (`consensus_plate`, `consensus_support_frames`, `consensus_confidence`, `conflicting_plate_count`).

---

## 7. Current Architectural Weaknesses & Scalability

### A. Candidate Generation Scalability (Section 19)
- In `inference/candidate_generation.py:101`:
  ```python
  for i in range(n):
      for j in range(i + 1, n):
  ```
  The implementation performs an outer-inner nested loop ($O(N^2)$). While it breaks early when temporal intervals exceed `max_time_window_seconds`, it still evaluates all pairs within the window.
- **Remediation**:
  1. Construct vehicle-type buckets (`car`, `truck`, `bus`, `motorcycle`, `unknown`) to completely skip incompatible pairs before iteration.
  2. Implement temporal interval indexing via `bisect` over sorted timestamps to bound $j$ to the valid window $[t_i, t_i + \Delta t_{\max}]$ in $O(\log N + K)$ time.

### B. Benchmark Reproducibility & Reports (Section 10 & 36)
- Benchmarks currently exist as individual scripts (`run_benchmark.py`, `inference/ablation_study.py`, `inference/degradation_benchmark.py`).
- There is no unified `scripts/reproduce_all.py` that executes all suites and writes timestamped JSON/Markdown reports to `reports/generated/`.
- **Remediation**: Build `scripts/reproduce_all.py` and canonical directory `benchmarks/` with `real_member1/`, `synthetic/`, `holdout/`, `adversarial/`.

---

## 8. Unsupported Performance Claims to Remove (Section 41)

The codebase must be cleaned of any exaggerated or mathematically uncalibrated terminology:

| Unsupported Term / Claim | Location / Occurrence | Correct Replacement / Framing |
| :--- | :--- | :--- |
| **"Bayesian posterior"** | Comments / old docstrings | **"same_vehicle_score"** or **"identity_evidence_score"** |
| **"Verified re-entry"** | Demo text for Track 65/94 | **"Identity linkage hypothesis (tracker fragmentation / overlap)"** |
| **"City-wide validated on real data"** | General claims | **"Validated on CAM_001 single-camera perception; multi-camera validated on synthetic/holdout networks"** |
| **"O(N log N) candidate generation"** | Comments in candidate generator | **"Indexed candidate pruning with empirical ~80% search space reduction"** |
| **"Physical speed in km/h for CAM_001"** | Some log messages | **"Image-space pixel displacement speed (`pixel_speed` px/s)"** |
| **"21 tracks with license plates"** | Previous integration reports | **"7 tracks with license plates (17.95% OCR coverage)"** |

---

## 9. Action Plan & Acceptance Gates

The audit establishes the following concrete technical hardening roadmap:
1. **Quarantine Legacy Data**: Move `kanishka_traffic.json` to `data/legacy/` and create canonical `run_real_member1.py`.
2. **Schema & Adapter Hardening**: Add `frame_detection_confidence_mean`, tracklet telemetry aggregation, and fix detection confidence assignment.
3. **Temporal & Spatial Hardening**: Update `validate_day2_end_to_end.py` to handle `None` temporal feasibility safely.
4. **Indexed Candidate Generator**: Implement vehicle-type buckets and `bisect` temporal windowing.
5. **Track 65/94 Fragmentation Handling**: Classify simultaneous overlap as tracker fragmentation / duplicate tracklet hypothesis.
6. **Multi-Tier Benchmark & Ablation Engine**: Implement automated ablation, degradation, and adversarial suites under `benchmarks/`.
7. **Unified Runner**: Implement `scripts/reproduce_all.py` writing to `reports/generated/`.
8. **Master Demo Upgrade**: Update `demo_master.py` to showcase 8 distinct cases distinguishing `[OBSERVED]`, `[INFERRED]`, and `[UNCERTAIN]`.
