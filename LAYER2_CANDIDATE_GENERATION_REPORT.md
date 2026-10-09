# UrbanTrack AI — Layer 2 Candidate Generation Report

**Stage**: Layer 2 Cross-Camera Association — Candidate Generation / Retrieval  
**Date**: October 8, 2026  
**Status**: READY  
**Test Suite**: 15/15 Candidate Generation Tests Passing (28/28 Layer 2 Tests Passing)  

---

## 1. Executive Summary & Architectural Scope

The **Layer 2 Candidate Generation** stage is strictly a **retrieval mechanism**. Its role is to take the frozen canonical tracklets produced by ingestion ($N = 7,448$) and extract physically plausible pairs of tracklets across camera boundaries for downstream multimodal identity fusion.

### Boundary Principles
- **No Identity Decisions**: Candidate generation determines whether a cross-camera transition is *physically plausible*. It does **NOT** decide whether two tracklets represent the same physical vehicle.
- **No Optimization / Clustering**: Hungarian assignment, graph partitioning, identity fusion, journey chaining, and route reconstruction are strictly forbidden at this boundary.
- **Layer 1 Immutability**: No files in `UrbanTrack_Member1_Handoff 2/` were modified.
- **High Recall Stance**: Retrieval operates conservatively. Ambiguous tracklets, missing embeddings, cross-architecture embeddings, and vehicle-type mismatches are preserved rather than filtered out.

---

## 2. Gating Architecture & Modules

The candidate generation pipeline is implemented in `layer2/candidate_generation/` across four modular components:

```
layer2/candidate_generation/
├── __init__.py               # Package exports
├── topology_gate.py          # Directed camera graph inspection (soft prior)
├── spatial_gate.py           # Haversine & graph distance; physical speed feasibility
├── temporal_gate.py          # Direction-independent chronological ordering & transit gating
└── candidate_generator.py    # Main pipeline, scenario partitioning, streaming serialization
```

### Module Responsibilities
1. **`TopologyGate`** (`topology_gate.py`):
   - Ingests `camera_graph.json` (65 camera nodes, 170 directed edges).
   - Functions as a **soft prior**: directed edges provide evidence of direct transit and road topology, but the absence of an edge **never** causes candidate rejection.
   - Strictly respects directed and one-way edges (e.g. `CAM_S02_C007 -> CAM_S02_C009` exists, but the reverse does not).

2. **`SpatialGate`** (`spatial_gate.py`):
   - Ingests `camera_locations.json` (WGS-84 coordinates for all 65 cameras).
   - Computes distance $d$ in meters: prefers `edge_distance_m` from `camera_graph.json` when a directed edge exists; falls back to Great-Circle Haversine distance ($R = 6,371,000.0\text{ m}$) otherwise.
   - Evaluates speed feasibility: $v = d / \Delta t$.

3. **`TemporalGate`** (`temporal_gate.py`):
   - Determines origin and destination tracklets strictly using synchronized start time (`start_sync_seconds`). Camera string identifiers are never used for ordering.
   - Computes physical transit gap: $\Delta t = t_{\text{dest, start}} - t_{\text{orig, end}}$.
   - Rejects negative travel times ($\Delta t < 0$) and zero travel times ($\Delta t = 0$).

4. **`CandidateGenerator`** (`candidate_generator.py`):
   - Enforces scenario partitioning (`NO_CROSS_SCENARIO_CANDIDATES`).
   - Filters same-camera pairs (`origin.camera_id == destination.camera_id`).
   - Gathers Re-ID compatibility, vehicle type agreement, and OCR availability metadata.
   - Streams JSON output directly to disk to prevent RAM blowup.

---

## 3. Exact Thresholds & Physical Rationale

| Parameter | Value | Unit | Physical & Operational Rationale |
| :--- | :---: | :---: | :--- |
| **Scenario Boundary** | Strict Isolation | — | CityFlow scenarios S01–S06 represent distinct geographic locations and non-overlapping recording times. Cross-scenario transit is physically impossible. |
| **Same-Camera Filter** | Exclusion | — | Layer 2 is dedicated to multi-camera cross-camera association. Intra-camera tracking is resolved by Layer 1 multi-object tracking. |
| **Chronological Ordering** | $\min(t_{\text{start, sync}})$ | seconds | Physical causality: vehicle departure must precede destination arrival. Strictly derived from clock-synchronized timestamps. |
| **Minimum Transit Gap ($\Delta t$)** | $> 0.0$ | seconds | Transit between distinct physical cameras ($d \ge 7.34\text{m}$) cannot occur instantaneously ($\Delta t = 0 \implies v = \infty$) or backwards in time ($\Delta t < 0$). |
| **Maximum Vehicle Speed ($v_{\max}$)** | $45.0$ | m/s | $45.0\text{ m/s} \approx 162.0\text{ km/h} \approx 100.7\text{ mph}$. Represents the maximum plausible physical velocity for urban arterials and ring roads. |
| **Minimum Vehicle Speed ($v_{\min}$)** | None (Soft) | m/s | Vehicles frequently stop at traffic signals (cycles up to 180s), sit in traffic congestion, or make brief stops. No minimum speed cutoff is enforced. |
| **Maximum Time Window ($T_{\max}$)** | None (Scenario-bounded) | seconds | CityFlow scenario recordings are short ($200\text{s}$ to $430\text{s}$). Arbitrary temporal truncations within a scenario are avoided. |

---

## 4. Empirical Evaluation & Pair Reduction

### Global vs Intra-Scenario Scale

$$\begin{aligned}
N_{\text{total tracklets}} &= 7,448 \\
\text{Theoretical Global All-Pairs} &= \frac{7,448 \times 7,447}{2} = \mathbf{27,732,628} \\
\text{Theoretical Intra-Scenario Pairs} &= \sum_{s \in \{S01\dots S06\}} \frac{N_s(N_s - 1)}{2} = \mathbf{9,160,912} \\
\text{Cross-Scenario Pairs Eliminated} &= 27,732,628 - 9,160,912 = \mathbf{18,571,716} \quad (66.97\%)
\end{aligned}$$

### Final Reduction Metrics

| Metric | Pair Count | Percentage |
| :--- | :---: | :---: |
| **Theoretical Global Pairs** | $27,732,628$ | $100.00\%$ |
| **Cross-Scenario Eliminated (Partitioning)** | $18,571,716$ | $66.97\%$ |
| **Intra-Scenario Pairs Evaluated** | $9,160,912$ | $33.03\%$ |
| **Intra-Scenario Pairs Rejected** | $2,297,286$ | $8.28\%$ of global ($25.08\%$ of intra-scenario) |
| **Valid Candidate Pairs Generated** | $\mathbf{6,863,626}$ | $\mathbf{24.75\%}$ of global ($\mathbf{74.92\%}$ of intra-scenario) |
| **Overall Global Reduction** | $\mathbf{20,869,002}$ | $\mathbf{75.25\%}$ |

---

## 5. Rejection Ledger Breakdown

Every rejected pair is serialized with machine-readable metadata in `results/layer2_candidates/candidate_rejections.json`.

| Reason Code | Count | Share of Rejections | Description |
| :--- | :---: | :---: | :--- |
| `REJECT_EXCESSIVE_SPEED` | $838,775$ | $36.51\%$ | Implied velocity $v = d / \Delta t > 45.0\text{ m/s}$ ($> 162\text{ km/h}$). |
| `REJECT_SAME_CAMERA` | $741,750$ | $32.29\%$ | Origin and destination tracklets share identical `camera_id`. |
| `REJECT_NEGATIVE_TIME` | $712,742$ | $31.03\%$ | Destination arrived before origin departed ($\Delta t < 0$). |
| `REJECT_ZERO_OR_INVALID_TIME` | $4,019$ | $0.17\%$ | Instantaneous departure and arrival ($\Delta t = 0.0\text{s}$) across non-zero camera distance. |
| **Total Rejections** | $\mathbf{2,297,286}$ | $\mathbf{100.00\%}$ | Complete machine-readable audit trail. |

---

## 6. Scenario-by-Scenario Breakdown

| Scenario | Cameras | Tracklets | Theoretical Pairs | Candidate Pairs | Rejected Pairs | Reduction | Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | 5 | 625 | 195,000 | 142,359 | 52,641 | 27.00% | 1.36s |
| **S02** | 4 | 730 | 266,085 | 184,454 | 81,631 | 30.68% | 1.67s |
| **S03** | 6 | 219 | 23,871 | 13,522 | 10,349 | 43.35% | 0.14s |
| **S04** | 25 | 781 | 304,590 | 234,818 | 69,772 | 22.91% | 2.01s |
| **S05** | 19 | 3,921 | 7,685,160 | 5,940,452 | 1,744,708 | 22.70% | 52.95s |
| **S06** | 6 | 1,172 | 686,206 | 348,021 | 338,185 | 49.28% | 4.47s |
| **Total** | **65** | **7,448** | **9,160,912** | **6,863,626** | **2,297,286** | **25.08%** | **62.59s** |

---

## 7. Subsystem Handling & Safety Invariants

### CAM_S01_C002 (MSMT17 Architecture Isolation)
- **Status**: `CAM_S01_C002` utilizes the `osnet_x0_25_msmt17` person Re-ID model, whereas all other 64 cameras utilize `osnet_x0_25_aicity`.
- **Handling**: In candidate records involving `CAM_S01_C002` paired with any AICity camera, the appearance block explicitly sets:
  ```json
  "appearance": {
    "origin_available": true,
    "destination_available": true,
    "compatible": false
  }
  ```
- **Safety Invariant**: Direct cosine similarity between MSMT17 and AICity vectors is mathematically prohibited. C002 candidate pairs are **preserved** (not rejected) so downstream fusion can associate them using spatial, temporal, motion, and license plate evidence.

### Missing Embeddings (868 Tracklets)
- **Status**: 868 canonical tracklets have `has_embedding == false` due to small bounding boxes or detector occlusion.
- **Handling**: In candidate records involving missing embeddings:
  ```json
  "appearance": {
    "origin_available": true,
    "destination_available": false,
    "compatible": false
  }
  ```
- **Safety Invariant**: Missing embeddings do **NOT** disqualify candidate pairs. They are preserved for downstream multimodal fusion.

### Vehicle Type Contradictions
- **Status**: Detectors exhibit class noise across views (e.g. car vs SUV vs truck).
- **Handling**: `vehicle_type.agreement` records `"EXACT_MATCH"` or `"MISMATCH"`.
- **Safety Invariant**: Class disagreement does not cause candidate rejection.

### Automatic Plate Recognition (ANPR / OCR)
- **Status**: 39,408 readable OCR readings exist across the dataset.
- **Handling**: Carries boolean flags `origin_available` and `destination_available`.
- **Safety Invariant**: No string matching or edit-distance filtering occurs at the candidate generation boundary.

---

## 8. Unit & Regression Test Verification

The test suite in `tests/layer2/test_candidate_generation.py` validates all 15 required behaviors:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/layer2/ -v
```

```
============================= test session starts ==============================
collected 28 items

tests/layer2/test_candidate_generation.py::test_cross_scenario_isolation PASSED [  3%]
tests/layer2/test_candidate_generation.py::test_same_camera_exclusion PASSED [  7%]
tests/layer2/test_candidate_generation.py::test_chronological_ordering PASSED [ 10%]
tests/layer2/test_candidate_generation.py::test_reverse_alphabetical_ordering_regression PASSED [ 14%]
tests/layer2/test_candidate_generation.py::test_negative_time_rejection PASSED [ 17%]
tests/layer2/test_candidate_generation.py::test_positive_time_acceptance PASSED [ 21%]
tests/layer2/test_candidate_generation.py::test_one_way_camera_edge_direction PASSED [ 25%]
tests/layer2/test_candidate_generation.py::test_no_edge_does_not_automatically_reject PASSED [ 28%]
tests/layer2/test_candidate_generation.py::test_excessive_speed_rejection PASSED [ 32%]
tests/layer2/test_candidate_generation.py::test_valid_speed_candidate PASSED [ 35%]
tests/layer2/test_candidate_generation.py::test_missing_embedding_handling PASSED [ 39%]
tests/layer2/test_candidate_generation.py::test_incompatible_reid_handling PASSED [ 42%]
tests/layer2/test_candidate_generation.py::test_vehicle_type_disagreement_tolerance PASSED [ 46%]
tests/layer2/test_candidate_generation.py::test_deterministic_output PASSED [ 50%]
tests/layer2/test_candidate_generation.py::test_rejection_reason_completeness PASSED [ 53%]
tests/layer2/test_layer2_ingestion.py::test_timestamp_parsing PASSED     [ 57%]
tests/layer2/test_layer2_ingestion.py::test_official_offset_application PASSED [ 60%]
tests/layer2/test_layer2_ingestion.py::test_8_fps_handling PASSED        [ 64%]
tests/layer2/test_layer2_ingestion.py::test_ocr_normalization PASSED     [ 67%]
tests/layer2/test_layer2_ingestion.py::test_ocr_aggregation_consensus PASSED [ 71%]
tests/layer2/test_layer2_ingestion.py::test_missing_ocr PASSED           [ 75%]
tests/layer2/test_layer2_ingestion.py::test_missing_embedding PASSED     [ 78%]
tests/layer2/test_layer2_ingestion.py::test_reid_compatibility PASSED    [ 82%]
tests/layer2/test_layer2_ingestion.py::test_scenario_isolation_validation PASSED [ 85%]
tests/layer2/test_layer2_ingestion.py::test_chronological_ordering_forward PASSED [ 89%]
tests/layer2/test_layer2_ingestion.py::test_reverse_camera_ordering_regression PASSED [ 92%]
tests/layer2/test_layer2_ingestion.py::test_duplicate_track_detection PASSED [ 96%]
tests/layer2/test_layer2_ingestion.py::test_embedding_normalization_check PASSED [100%]

============================== 28 passed in 0.03s ==============================
```

### Key Regression Test Passed
- **Historical Reverse-Direction Defect**: When a vehicle moves from `CAM_S01_C002` to `CAM_S01_C001`, lexicographical camera sorting previously set $C001$ as origin and $C002$ as destination, resulting in negative transit time and a false rejection. Test `test_reverse_alphabetical_ordering_regression` confirms that `start_sync_seconds` dictates origin/destination, computing a valid positive $\Delta t = +4.0\text{s}$ and preserving the candidate.

---

## 9. Performance & Artifact Inventory

### System Performance
- **Wallclock Runtime**: $63.13$ seconds for full end-to-end processing of 7,448 tracklets across 9.16M pair evaluations.
- **Peak Memory**: $326.06\text{ MB}$ (achieved via generator streaming directly into formatted JSON array files).

### Generated Artifacts in `results/layer2_candidates/`

1. **`candidate_pairs.json`** ($5.2\text{ GB}$):
   - $6,863,626$ valid, physically plausible cross-camera candidate pairs.
   - Conforms strictly to the frozen Layer 2 candidate schema.
2. **`candidate_rejections.json`** ($707\text{ MB}$):
   - $2,297,286$ machine-readable rejection records with `candidate_pair_id`, `reason_code`, and `reason_detail`.
3. **`candidate_summary.json`** ($4.3\text{ KB}$):
   - Complete execution metadata, theoretical pair counts, candidate counts, reduction percentages, and subsystem handling audits.
4. **`candidate_validation.json`** ($1.7\text{ KB}$):
   - Formal verification report certifying all 10 core safety invariants (`overall_status: "PASS"`).

---

## 10. Recall Statement & Limitations

### Candidate Recall Statement
> **CANDIDATE RECALL NOT MEASURED**
> 
> In accordance with scientific measurement protocol, candidate recall is **not** claimed or fabricated at this stage because true cross-camera vehicle identity evaluation belongs strictly to the downstream evaluation stage where ground truth identities are loaded and matched.

### Known Limitations
1. **Straight-Line / Haversine Distance Fallback**: When two cameras do not share a directed edge in `camera_graph.json`, great-circle Haversine distance is used as a lower bound for road distance. This is conservative and safe for retrieval.
2. **Signal Delay Variance**: Traffic signal waiting times vary from 30s to 180s. Because minimum velocity is not hard-gated, candidates with long delays are preserved, which maintains high retrieval recall but yields a large candidate set in dense scenarios (S05).

---

## Final Status Verdict

```
LAYER2_CANDIDATE_GENERATION = READY
```
