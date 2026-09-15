# UrbanTrack AI — Member 2 Canonical Architecture & Reasoning Engine

**Document Version**: 1.1.0 (Final Hardened Production)  
**Author / Responsibility**: Member 2 (Vivek) — Candidate Generation, Identity Fusion, Identity Graph, Trajectory Inference, Benchmarking & Evaluation  
**Status**: Verified & Auditable  

---

## 1. Canonical Production Reasoning Pipeline

The Member 2 subsystem implements a single, strictly defined canonical production inference path. There are no dual implementations, heuristic shortcuts, or benchmark-only bypasses. Every benchmark, test, and production demo executes this identical pipeline:

```
                  Raw Observation Feed (Member 1 / Sensors)
                                     │
                                     ▼
                    Observation Ingestion & Validation
                       (schemas/observation_schema.py)
                                     │
                                     ▼
                        Candidate Pair Generation
                   (inference/candidate_generation.py)
                                     │
                                     ▼
                       Multimodal Identity Fusion
                     (inference/identity_fusion.py)
                                     │
                                     ▼
                    Identity Graph Assembly & Clustering
                      (inference/identity_graph.py)
                                     │
                                     ▼
                    Global Identity Cluster Hypotheses
                                     │
                                     ▼
                    Trajectory Inference & Gap Routing
                     (inference/trajectory_engine.py)
                                     │
                                     ▼
               Evidence Provenance & Route Uncertainty Ledger
```

---

## 2. Component Specifications

### 2.1 Observation Ingestion & Contract Enforcement (`schemas/observation_schema.py`)
- **Purpose**: Defines the immutable data contract between perception sensors (Member 1) and downstream reasoning engines (Member 2).
- **Schema Contracts**:
  - `observation_id`: Globally unique identifier (`OBS_...`).
  - `camera_id`: Camera sensor identifier.
  - `timestamp_seconds`: Continuous floating-point second timestamp.
  - `timestamp_semantics`: Strictly isolated as `"video_relative"` (local frame offset) or `"synchronized"` (NTP/PTP reference).
  - `point_coordinate_system`: Strictly isolated as `"image"` (pixels) or `"gps"` (WGS-84 decimal degrees).
  - `appearance_embedding`: 512-dimensional L2-normalized float vector (OSNet MSMT17). Validated for non-emptiness, finiteness (no NaN/Inf), and dimensional consistency.
  - `plate` & `plate_confidence`: Alphanumeric license plate string and verified OCR confidence in $[0.0, 1.0]$.
  - `camera_reliability`: Dynamic sensor health metric in $[0.0, 1.0]$.

### 2.2 Candidate Generation Engine (`inference/candidate_generation.py`)
- **Purpose**: Drastically reduces the $O(N^2)$ theoretical observation pair space to a sub-linear candidate set without pruning true same-vehicle matches.
- **Filtering Mechanisms**:
  1. **Temporal Horizon Pruning**: Observations sorted chronologically; bisect-right binary search limits lookups to $t \in [t_a, t_a + \Delta t_{\max}]$ ($\Delta t_{\max} = 1800	ext{s}$).
  2. **Vehicle Type Incompatibility**: Discards pairs where both vehicle types are known and incompatible (e.g. `car` vs `motorcycle`, `bus` vs `car`).
  3. **Simultaneous Cross-Camera Pruning**: If two observations share the exact same timestamp on different cameras with synchronized clocks ($\Delta t = 0$), they are physically impossible and pruned.
  4. **Kinematic Speed Limit Bounds**: Discards pairs requiring travel speed $> 120	ext{ km/h}$ over geographic distance $D(A, B)$.
- **Complexity Analysis**:
  - **Average Case**: $O(N \log N)$ when observations are temporally dispersed across an operational timeline, leveraging bisect-right searches over chronological observation arrays.
  - **Worst Case**: $O(N^2)$ if all $N$ observations fall within the exact same temporal horizon $[t, t + \Delta t_{\max}]$ and share identical or wildcard vehicle types. UrbanTrack AI explicitly acknowledges this quadratic worst-case bound without claiming artificial $O(N \log N)$ guarantees.
- **Diagnostic Ledger**: Maintains exact counters for `temporal_horizon_exceeded`, `incompatible_vehicle_type`, `simultaneous_different_cameras`, and `physically_impossible_speed`.

### 2.3 Multimodal Identity Fusion Engine (`inference/identity_fusion.py`)
- **Purpose**: Evaluates candidate pairs across multiple heterogeneous modalities and produces an audit-traceable `same_vehicle_score` and `decision_state`.
- **Modality Handlers**:
  - **Appearance (OSNet 512-D)**: Cosine similarity $\cos(u, v) = rac{u \cdot v}{\|u\|_2 \|v\|_2} \in [0.0, 1.0]$.
  - **License Plate (OCR)**: Normalized Levenshtein edit distance similarity.
  - **Adaptive Modality Weighting**: When both plate and appearance are available, weights adapt dynamically to verified OCR confidence:
    $$w_{	ext{plate}} = 0.30 + 0.30 	imes c_{	ext{ocr}}, \quad w_{	ext{app}} = 1.0 - w_{	ext{plate}}$$
  - **Spatio-Temporal Feasibility**: Gating factor combining kinematics and vehicle-type compatibility:
    $$S_{	ext{feasibility}} = 0.70 	imes S_{	ext{spatiotemporal}} + 0.30 	imes S_{	ext{type}}$$
- **Contradiction Detection**:
  - Vehicle type mismatch $\implies 	ext{Score} = 0.0$, `REJECTED`.
  - Physically impossible speed ($> 120	ext{ km/h}$) $\implies 	ext{Score} = 0.0$, `REJECTED`.
  - Strong plate contradiction (similarity $< 0.35$ with verified OCR confidence $\ge 0.50$) $\implies 	ext{Score} = 0.0$, `REJECTED`.
- **Decision States**:
  - `CONFIRMED`: Score $\ge 	au_{	ext{confirmed}}$ ($0.75$) with positive identity evidence.
  - `AMBIGUOUS`: Score $\in [0.40, 0.75)$ or partial conflicting evidence.
  - `REJECTED`: Score $< 0.40$ or physical contradiction detected.

### 2.4 Identity Graph Engine (`inference/identity_graph.py`)
- **Purpose**: Assembles pairwise edges into a unified graph, validates cluster physical consistency, resolves transitive contradictions, and extracts global vehicle identity hypotheses.
- **Key Operations**:
  1. **Edge Formation**: Edges formed only if `score` $\ge 	au_{	ext{min}}$ ($0.70$) AND positive identity evidence is available.
  2. **Transitive Contradiction Detection**: Detects cases where $A \sim B$ and $B \sim C$, but $A$ and $C$ exhibit a physical or plate contradiction.
  3. **Cluster Consistency Validation**: Verifies that chronologically ordered cluster members do not require impossible speeds or simultaneous presence at different cameras.
  4. **Contradiction Splitting**: Automatically partitions contradictory clusters using connected components along non-contradictory spanning trees.
  5. **Explain Non-Merge Ledger**: Caches rejection reasons (`rejection_records`) explaining why any two observations were not merged into the same identity.

### 2.5 Trajectory & Route Uncertainty Engine (`inference/trajectory_engine.py` & `inference/sparse_engine.py`)
- **Purpose**: Reconstructs physical mobility corridors across road networks and quantifies route ambiguity over unobserved camera gaps.
- **Mechanics**:
  - Models urban road topologies as directed graphs with nodes (junctions) and edges (road segments).
  - Evaluates candidate paths between consecutive camera sightings using speed limits and expected transit times.
  - Computes relative route likelihoods and quantifies route entropy ($H = -\sum p_i \ln p_i 	ext{ nats}$).
  - **Zero Fabrication Policy**: Gaps between cameras are represented as candidate route distributions, never as fabricated intermediate camera detections.

---

## 3. Provenance & Explainability

Every identity decision produces an audit ledger containing:
```json
{
  "source_observation": "OBS_MC_00102",
  "target_observation": "OBS_MC_00103",
  "same_vehicle_score": 0.9754,
  "decision_state": "CONFIRMED",
  "evidence": {
    "appearance_similarity": 0.9412,
    "plate_similarity": 1.0,
    "ocr_confidence": 0.95,
    "spatial_feasibility": 1.0,
    "temporal_feasibility": 1.0,
    "vehicle_type_status": "compatible"
  },
  "explanation": "Estimated match probability is 0.98: Appearance similarity is 0.94 (high), plate exact match (1.00) with 0.95 OCR confidence."
}
```
