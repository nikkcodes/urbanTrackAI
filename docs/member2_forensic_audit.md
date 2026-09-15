# UrbanTrack AI — Member 2 Forensic Technical Audit

**Audit Date**: September 2026  
**Auditor**: Senior ML Systems Engineer (Member 2 — Reasoning, Cross-Camera Fusion, and Benchmarking)  
**Target Repository**: `urbantrack-ai`  
**Scope**: Complete forensic audit of Member 2 reasoning components, benchmark pipelines, production contracts, and data boundaries prior to independent multi-camera benchmark implementation.

---

## 1. Executive Summary

This forensic audit analyzes the executable state of the UrbanTrack AI Member 2 codebase. UrbanTrack AI is a probabilistic city-scale mobility intelligence engine designed to process camera perception outputs (bounding boxes, tracklets, OCR license plates, OSNet 512-D appearance embeddings, and camera telemetry) and perform cross-camera identity fusion, contradiction-aware identity graph clustering, and multi-hypothesis trajectory inference.

Member 2 is responsible for:
1. Candidate generation (spatial-temporal indexing, vehicle-type partitioning)
2. Identity evidence fusion (multimodal evidence ledger, contradiction vetoing)
3. Contradiction-aware IdentityGraph (disjoint-set clustering with mutual exclusion checks)
4. Trajectory hypothesis engine (sparse network gap inference, Shannon entropy)
5. Validation, degradation, scalability, and reproduction pipelines

Prior implementation passes verified basic unit tests and canonical Member 1 perception ingestion. However, expert jury review requires:
- An independent multi-camera benchmark (`multicamera_v1`) where ground truth is generated from latent vehicle identities strictly **before** observation generation,
- Realistic difficulty tiers (EASY, MEDIUM, HARD, ADVERSARIAL) with hard negatives,
- Empirical OSNet 512-D embedding distribution usage,
- Absolute elimination of self-assigned scores or claims of calibrated probabilities without calibration evidence,
- Clear delineation between single-camera perception validation (`REAL_MEMBER1_CAM_001`) and multi-camera reasoning benchmarks.

---

## 2. Current Architecture & Production Execution Path

### Canonical Member 2 Production Pipeline
The system enforces a single, authoritative execution path for both live inference and benchmark evaluation:

```
[Camera Perception Feeds / Tracklets]
                 ↓
      Observation Validation (Schema Contract)
                 ↓
      CandidateGenerator (Spatio-Temporal & Type Gating)
                 ↓
      IdentityFusion (Multimodal Evidence Ledger + Contradiction Veto)
                 ↓
      IdentityGraph (Disjoint-Set Union-Find + Contradiction Validation)
                 ↓
      Global Identity Clusters
                 ↓
      Trajectory Hypothesis Engine (Road Graph Corridors + Shannon Entropy)
                 ↓
      Downstream Handoff (Mobility Conservation & Privacy Guard)
```

### Module Responsibilities:
1. **`schemas/observation_schema.py`**:
   - Strictly enforces observation contracts: `image_space_trajectory_point` (pixels) vs physical GPS coordinates, `video_relative` elapsed seconds vs synchronized wall-clock time, detector confidence vs OCR confidence vs camera reliability.
2. **`inference/candidate_generation.py`**:
   - Implements `CandidateGenerator`: bisect-sorted temporal indexing over time windows ($\Delta t \le \tau_{\max}$), strict vehicle-type compatibility partitioning, and spatial bounding.
   - Evaluates candidate reduction and candidate recall without duplicate pairwise generation.
3. **`inference/similarity.py`**:
   - Computes $L_2$-normalized OSNet appearance cosine similarity, Jaro-Winkler character distance for license plates with character ambiguity maps (e.g., 0/O, 8/B, 1/I), and validates finite vectors.
4. **`inference/temporal.py` & `inference/spatial.py`**:
   - Evaluates kinematic feasibility: maximum plausible velocity threshold ($v_{\max} = 120\text{ km/h}$), impossible negative time, simultaneous occupancy conflicts.
5. **`inference/identity_fusion.py`**:
   - Evaluates multilateral evidence, producing an explicit evidence ledger distinguishing: `available_supportive`, `available_contradictory`, `missing`, `unavailable`, and `not_applicable`.
   - Vetoes candidate matches on physical contradictions (e.g. impossible travel speed, conflicting confirmed plates, simultaneous occupancy).
   - Outputs uncalibrated `same_vehicle_score` with three explicit decision states: `CONFIRMED`, `AMBIGUOUS`, `REJECTED`.
6. **`inference/identity_graph.py`**:
   - Disjoint-set graph clusterer that actively invokes `CandidateGenerator` inside `build_graph()`.
   - Re-evaluates all intra-cluster edges to detect and split contradictory components.
7. **`inference/sparse_engine.py` & `inference/trajectory_engine.py`**:
   - Performs route corridor inference across physical road networks using Dijkstra shortest paths, computing relative route likelihoods and Shannon entropy ($H$). Never fabricates intermediate camera observations.

---

## 3. Current Benchmark Paths

Currently, the repository contains the following benchmark modules:
1. **`inference/ablation_study.py`**:
   - Evaluates 6 isolated tiers ($A \to F$: Re-ID only, Plate only, Re-ID+Plate, +Temporal, +Spatial, Full UrbanTrack) with explicit coverage metrics and zero silent fallback.
2. **`inference/holdout_benchmark.py`**:
   - Splits a controlled dataset into Development and Holdout splits; tunes decision threshold $\tau^*$ exclusively on Dev, freezes $\tau^*$, and evaluates Holdout once.
3. **`inference/degradation_benchmark.py`**:
   - Sweeps plate dropout (0–100%), Re-ID dropout (0–100%), and camera reliability attenuation (1.0 down to 0.0), dynamically computing the maximum false merge rate.
4. **`inference/adversarial_suite.py`**:
   - 16 deterministic edge cases (`ADV_01` to `ADV_16`) testing tracker fragmentation, identical vehicles, impossible speeds, duplicate detections, and sensor failures.
5. **`scripts/reproduce_all.py`**:
   - Master reproduction script orchestrating integrity verification, contract checks, ablation, holdout evaluation, fair scalability benchmarking, degradation, and adversarial validation.

---

## 4. Current Test Status

Execution of Python's standard unittest discovery:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
- **Total Test Count**: 350 tests across 20 test modules
- **Passing**: 350 / 350 (100%)
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: ~18.6 seconds

---

## 5. Current Data Limitations & Methodological Weaknesses

### 1. Single-Camera Scope of Real Data (`REAL_MEMBER1_CAM_001`)
- The authoritative raw perception data from Member 1 represents a single CCTV camera (`CAM_001`, `traffics.mp4`, 613 frames, 4,821 YOLOv8 detections, 39 tracklets).
- **Limitation**: In a single-camera feed, no true cross-camera positive identity pairs exist. All true matches are identity loops (tracklet self-matches or tracker fragmentation like Track 65/94).
- **Methodological Rule**: The real-data path validates perception integration, schema parsing, feature normalization, and same-camera contradiction handling. It MUST NOT claim city-scale multi-camera identity recall.

### 2. Previous Synthetic Benchmark Scale & Modularity
- While `inference/holdout_benchmark.py` implemented a controlled 24-vehicle generator, the repository lacked a dedicated, modular multi-camera benchmark package modeling 4–6 cameras, 100–200 latent vehicles, and 1,000–3,000 observations with independently stored ground truth and pairwise labels.
- The benchmark needs difficulty tiers (EASY, MEDIUM, HARD, ADVERSARIAL) and hard negatives (different vehicles sharing visual or temporal proximity).

### 3. OSNet Embedding Distribution in Synthetic Data
- Some synthetic generation routines used random low-dimensional or synthetic Gaussian vectors.
- **Requirement**: Benchmark embeddings must be calibrated to the actual empirical 512-D OSNet embedding distribution present in `data/member1_perception/cam_001/track_embeddings.json`.

### 4. Score Terminology & Probability Claims
- Historical references in legacy code used `same_vehicle_probability`.
- **Methodological Rule**: Because the fusion output is an uncalibrated heuristic score $[0.0, 1.0]$, it is exposed as `same_vehicle_score`. Uncalibrated probabilities or Bayesian posteriors are strictly disclaimed.

---

## 6. Files That Will Be Added or Modified

### New Benchmark Infrastructure Package (`inference/benchmark/`):
- `inference/benchmark/__init__.py`: Package exports.
- `inference/benchmark/difficulty.py`: Difficulty tier models (EASY 30%, MEDIUM 40%, HARD 20%, ADVERSARIAL 10%) with explicit noise generators (OCR character confusion, feature jitter, timing noise).
- `inference/benchmark/ground_truth.py`: Latent vehicle identity model, vehicle generator, and independent pairwise ground-truth matrix.
- `inference/benchmark/generator.py`: Multi-camera observation generator (5 cameras, ~150 latent vehicles, ~1,500-2,000 observations) with realistic camera transition topology, hard negative generation, and empirical OSNet 512-D embedding sampling.
- `inference/benchmark/evaluator.py`: Standardized evaluation harness computing pairwise TP/FP/TN/FN, precision, recall, F1, false merge rate, false split rate, and difficulty tier breakdowns.
- `inference/benchmark/runner.py`: Executable runner script for standalone benchmark generation, execution, and evaluation.

### New Benchmark Data Assets (`data/benchmarks/multicamera_v1/`):
- `data/benchmarks/multicamera_v1/cameras.json`: Camera coordinates, topologies, and reliability scores.
- `data/benchmarks/multicamera_v1/observations.json`: Schema-compliant observation records.
- `data/benchmarks/multicamera_v1/ground_truth.json`: True latent vehicle assignments for each observation.
- `data/benchmarks/multicamera_v1/pairwise_ground_truth.json`: Independent pairwise labels with difficulty tags.
- `data/benchmarks/multicamera_v1/metadata.json`: Dataset metadata declaring `dataset_type: "synthetic_controlled"`.

### Updated Documentation Deliverables (`docs/`):
- `docs/member2_forensic_audit.md` (this file)
- `docs/member2_architecture.md`: Canonical production reasoning path specification.
- `docs/member2_benchmark_methodology.md`: Rigorous benchmark protocol, independent ground truth structure, difficulty tiers, and evaluation metrics.
- `docs/member2_data_semantics.md`: Complete data dictionary, schema contracts, and score terminology.
- `docs/member2_change_log.md`: Detailed changelog of all additions and rationale.

### Updated Scripts and Tests:
- `scripts/reproduce_all.py`: Integrated independent multi-camera benchmark stage.
- `tests/test_benchmark_multicamera.py`: Regression and unit tests for the benchmark suite.
- `reports/generated/final_validation.json` & `reports/generated/final_technical_audit.md`: Updated reports.

---

## 7. Files That Must NOT Be Modified

1. **Member 1 Raw Assets**:
   - `data/member1_perception/cam_001/raw_frame_detections.json`
   - `data/member1_perception/cam_001/track_embeddings.json`
   - `data/member1_perception/cam_001/camera_telemetry.json`
   - `data/member1_perception/cam_001/manifest.json`
   - Must remain byte-for-byte untouched to maintain cryptographic SHA-256 verification.
2. **Member 3 Adapters & Downstream Packages**:
   - `mobility/`, `anomaly/`, `simulation/`: Out of Member 2 ownership scope; preserve existing interfaces.
3. **Core Public Schemas**:
   - `schemas/observation_schema.py`: Keep public fields and validations intact.

---

## 8. Implementation Risks & Mitigation Strategy

| Risk | Potential Impact | Mitigation Strategy |
|---|---|---|
| Benchmark generation introduces non-deterministic test failures | Flaky CI / evaluation | Fix global random seeds (`random.seed(42)`, `np.random.seed(42)`) across all generators. |
| Hard negatives cause excessive false merges | Degraded benchmark precision | Tune multimodal contradiction handling to ensure conservative `AMBIGUOUS` decisions instead of false `CONFIRMED`. |
| Excessive benchmark scale causes slow execution in test suite | Timeout in `unittest` | Benchmark generator supports configurable vehicle/observation scales; unit tests run on compact fixtures ($N=20$), while full benchmark runs $N=150$ vehicles. |
| Schema mismatch with `Observation` dataclass | Ingestion crashes | Validate all generated observations through `Observation` constructor and schema validator. |
