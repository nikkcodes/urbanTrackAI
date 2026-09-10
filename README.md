# UrbanTrack AI — Mobility Inference Engine (Days 1, 2, 3 & 4 Complete)

City-scale multi-camera vehicle identity fusion, probabilistic trajectory reconstruction, and sparse missing-camera trajectory inference for **UrbanTrack AI**.

**Role**: Vivek — Mobility Inference Engineer (Member 2)  
**Scope**: End-to-end pipeline from perception ingestion (Day 1) to cross-camera identity fusion (Day 2), road-network trajectory reconstruction (Day 3), and sparse missing-camera hidden route inference (Day 4).

---

# Day 1: System Foundation & Schema Standardization (Day 1 Complete)

Foundation layer establishing observation data structures, camera coordinate mapping, spatio-temporal distance metrics, and similarity primitives.

### Day 1 Deliverables & Architecture:
- **Observation Schema ([schemas/observation_schema.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/schemas/observation_schema.py)):** Standardized `Observation` dataclass supporting detection confidence, bounding boxes, variable-dimension appearance embeddings, license plates, and geographic coordinates.
- **Camera Metadata Loader ([inference/observation_loader.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/observation_loader.py)):** Ingestion of camera locations ([data/cameras/camera_metadata.json](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/data/cameras/camera_metadata.json)) and perception feeds.
- **Similarity & Distance Metrics ([inference/similarity.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/similarity.py)):**
  - Appearance cosine similarity with dimension validation.
  - License plate Levenshtein similarity with OCR error tolerance.
  - Vehicle type compatibility matrix.
  - Geographic Haversine distance.
  - Epoch timestamp differences.
- **Spatio-Temporal Feasibility Primitives:**
  - [inference/spatial.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/spatial.py): Travel speed validation against city limits.
  - [inference/temporal.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/temporal.py): Temporal order validation ($\Delta t \ge 0$).
- **Day 1 Verification & Demo:**
  - [demo.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/demo.py): End-to-end Day 1 demonstration script on sample feeds.
  - [tests/test_observation.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/tests/test_observation.py): Schema and serialization unit tests.
  - [tests/test_similarity.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/tests/test_similarity.py): Unit tests for all similarity and distance functions.

---

# Day 2: Identity Fusion Engine & Identity Graph (Day 2 Complete)

Core cross-camera vehicle identity matcher answering:
> *"Given two vehicle observations from different cameras, how likely is it that they represent the same physical vehicle?"*

### Day 2 Architecture & Pipeline Flow:
```text
Kanishka Perception Feed (JSON)
           ↓
    Observation Loading (observation_loader.py)
           ↓
 Pairwise Feasibility & Identity Scoring (identity_fusion.py)
           ↓
    Identity Graph Construction (identity_graph.py)
           ↓
 Candidate Identity Clustering (Day 3 Contract Output)
```

---

## Day 2 Input Schemas

### 1. Vehicle Observation Input
`schemas/observation_schema.py` standardizes vehicle perception feeds:
- `observation_id` (str): Unique identifier.
- `camera_id` (str): Unique camera identifier.
- `timestamp_seconds` (float): Epoch timestamp in seconds.
- `frame_id` (int): Frame index.
- `track_id` (str): Local perception track identifier.
- `vehicle_type` (str): Categorical vehicle type (`sedan`, `car`, `bus`, `truck`, `auto`, etc.).
- `detection_confidence` (float): Perception confidence in `[0.0, 1.0]`.
- `bbox` (list[float]): Bounding box `[x1, y1, x2, y2]`.
- `appearance_embedding` (list[float] | None): Variable-dimension Re-ID feature vector.
- `plate` (str | None): License plate text.
- `latitude` / `longitude` (float | None): Geographic coordinates.

### 2. Camera Metadata Input
`data/cameras/camera_metadata.json` maps `camera_id` to geographic coordinates:
```json
{
  "cam_01": {
    "latitude": 17.3850,
    "longitude": 78.4867
  }
}
```

---

## Pairwise Scoring & Identity Evidence

Pairwise match probabilities $P_{\text{match}}$ are computed in `inference/identity_fusion.py`:

1. **Hard Physical Rejections ($P_{\text{match}} = 0.0$):**
   - **Incompatible Vehicle Types:** E.g., `car` vs `bus`.
   - **Chronologically Inverted Timestamps:** $\Delta t < 0$ (Observation B occurs before A).
   - **Simultaneous Different Cameras:** $\Delta t = 0$ at two distinct physical cameras.
   - **Simultaneous Same Camera:** $\Delta t = 0$, same frame ID, distinct bounding boxes.
   - **Physically Impossible Speed:** Travel speed $V_{\text{req}} = \frac{D}{\Delta t} \times 3.6 > 120$ km/h.

2. **Feasibility Gating Factor ($S_{\text{feasibility}}$):**
   - $S_{\text{ST}} = \frac{S_{\text{temporal}} + S_{\text{spatial}}}{2}$
   - $S_{\text{feasibility}} = 0.70 \cdot S_{\text{ST}} + 0.30 \cdot S_{\text{type}}$

3. **Available Identity Evidence ($S_{\text{ID}}$):**
   - Combines normalized Cosine Similarity of Re-ID vectors ($S_{\text{app}}$) and Levenshtein similarity of plates ($S_{\text{plate}}$).
   - If **identity evidence is present**:  
     $P_{\text{match}} = S_{\text{feasibility}} \cdot S_{\text{ID}}$
   - If **identity evidence is missing or invalid**:  
     $P_{\text{match}} = S_{\text{feasibility}} \cdot 0.50$ (Capped at unconfirmed score $0.50$, marked `appearance_status: "missing"`).

---

## Data Handling & Fallback Behavior

- **Missing Appearance Embedding (`None`):** Treated as unavailable evidence ($P = 0.50$, `appearance_status: "missing"`). Does **NOT** produce $P = 1.0$.
- **Invalid / Mismatched Dimensions:** `appearance_similarity` returns `None`. Sets `appearance_status: "invalid_mismatched"`. Pipeline executes safely without crashing.
- **Missing Coordinates:** Defaults spatial feasibility to neutral prior ($0.50$).

---

## Identity Graph & Clustering Safety

`inference/identity_graph.py` constructs a graph of observations:
- **Default Threshold:** Configurable `min_probability_threshold = 0.70` (requiring confident identity match evidence).
- **Edge Creation Rule:** Graph edges are formed **only** when $P \ge 0.70$ AND positive identity evidence is available (`identity_evidence_available == True`).
- **Safety:** Prevents ambiguous or unconfirmed observations ($P = 0.50$) from forming graph edges and creating transitive over-clusters.

---

## Day 2 Output Contract (Day 3 Handoff Interface)

Candidate vehicle identity output contract exported by `IdentityGraph`:

```json
{
  "identity_id": "VEHICLE_CANDIDATE_001",
  "member_observations": [
    {
      "observation_id": "obs_001",
      "camera_id": "cam_01",
      "timestamp_seconds": 10.0,
      "latitude": 17.3850,
      "longitude": 78.4867,
      "vehicle_type": "car"
    },
    {
      "observation_id": "obs_002",
      "camera_id": "cam_02",
      "timestamp_seconds": 40.0,
      "latitude": 17.3880,
      "longitude": 78.4867,
      "vehicle_type": "car"
    }
  ],
  "identity_confidence": 0.9998,
  "identity_evidence_summary": {
    "appearance": "available",
    "temporal": "supported",
    "spatial": "supported",
    "vehicle_type": "compatible"
  }
}
```

---

# Day 3: Probabilistic Trajectory Reconstruction (Day 3 Complete)

Cross-camera vehicle trajectory inference engine across the physical urban road network for **UrbanTrack AI**.

**Role**: Vivek — Mobility Inference Engineer (Member 2)  
**Scope Boundary**: Candidate route generation, road-aware spatial/temporal feasibility, route scoring, ranked trajectory hypotheses, uncertainty preservation, and multi-observation trajectory reconstruction.  
*(Excludes: city-wide traffic analytics, OD matrix analytics, congestion prediction, counterfactual simulation, GIS dashboard, and LLM explanations).*

---

## Day 3 Objective

Answer the core trajectory inference question:
> *"Given that observations belonging to the same physical vehicle were detected at two different cameras/times, infer the most plausible route the vehicle could have taken between those observations across the physical road network."*

Crucially:
- Do **NOT** calculate a straight line between cameras.
- Do **NOT** automatically pick only the shortest path.
- Preserve plausible alternatives and routing uncertainty honestly.

---

## Day 3 Architecture & Pipeline Flow

```text
Day-2 Identity Clusters / Sequential Observations
                    +
      Spatial Road Graph (road_graph.py)
                    ↓
  Camera-to-Road Network Association (Snapping & Explicit Mapping)
                    ↓
      Candidate Route Generation (Top-K Loop-Free Path Search)
                    ↓
     Road-Aware Temporal & Spatial Feasibility Checking
                    ↓
     Route Likelihood Scoring (Deterministic & Explainable)
                    ↓
     Trajectory Hypothesis Ranking & Uncertainty Preservation
                    ↓
   Complete Multi-Observation Vehicle Trajectory (VehicleTrajectory)
```

---

## Road Graph Interface (`RoadGraph`)

The trajectory engine interacts with a directed/bidirectional spatial network stored in `data/roads/synthetic_road_graph.json` or external graph feeds:

```json
{
  "nodes": [
    {
      "id": "junc_01",
      "name": "Charminar Junction",
      "latitude": 17.3850,
      "longitude": 78.4867
    }
  ],
  "edges": [
    {
      "road_id": "road_01",
      "name": "Patharghatti Rd",
      "from": "junc_01",
      "to": "junc_02",
      "distance_m": 414.8,
      "speed_limit_kmh": 50.0,
      "expected_speed_kmh": 35.0,
      "one_way": false
    }
  ],
  "camera_associations": {
    "cam_01": "junc_01"
  }
}
```

### Camera-to-Road Network Association
- **Explicit Association:** Direct mapping (`camera_associations[cam_id] = node_id`).
- **Spatial Snapping:** Computes Haversine distance from camera coordinates to nearest junction node. Snaps only if distance $\le 150.0\text{m}$.
- **Rejection Safety:** If camera coordinates are absent or exceed the maximum snapping distance, returns `None` and marks status `"unassociated_camera"`. Does **not** fabricate non-existent routes.

---

## Candidate Route Generation (Top-K)

- Implements Dijkstra's algorithm for true shortest path distance.
- Implements bounded loop-free alternative path exploration:
  - Generates top-K alternative routes (default $K = 5$).
  - Bounds path search depth (`max_depth = 10`) and distance threshold ($2.0 \times \text{shortest\_distance}$).
  - Explores distinct physical corridors (e.g. urban commercial street vs bypass expressway) rather than collapsing to a single route.

---

## Road-Aware Feasibility & Scoring

For each candidate route, the engine evaluates:
1. **Route Distance ($D$):** Sum of segment lengths.
2. **Observed Time Window ($\Delta t$):** $t_{\text{end}} - t_{\text{start}}$.
3. **Required Travel Speed ($V_{\text{req}}$):** $\frac{D}{\Delta t} \times 3.6$ km/h.
4. **Road-Specific Speed Feasibility:**
   - Compares $V_{\text{req}}$ against route speed limits $V_{\text{limit}}$ (with a configurable $25\%$ tolerance factor for congestion-free burst travel).
   - Routes exceeding $V_{\text{limit}} \times 1.25$ or the physical vehicle cap ($120\text{ km/h}$) are strictly marked `feasible: false` and penalized.
5. **Route Scoring ($S_{\text{raw}}$):**
   - Speed compatibility score $S_{\text{speed}} \in [0, 1]$ based on expected road flow.
   - Distance efficiency score $S_{\text{dist}} = \frac{D_{\text{shortest}}}{D_{\text{route}}} \in [0, 1]$.
   - Composite raw score: $S_{\text{raw}} = 0.65 \cdot S_{\text{speed}} + 0.35 \cdot S_{\text{dist}}$.
6. **Relative Estimated Likelihood:**
   - Normalizes raw scores across all feasible candidates:
     $$P_i = \frac{S_{\text{raw}, i}}{\sum_j S_{\text{raw}, j}}$$
   - **Documented Principle:** These are relative estimated likelihoods, **not** statistically calibrated Bayesian posteriors.

---

## Uncertainty Preservation

- If top alternative routes have comparable feasibility:
  $$|P_1 - P_2| < \text{ambiguity\_threshold}\ (0.15)$$
- The system flags `is_ambiguous = true` and generates an explicit explanation:
  `"Trajectory is ambiguous between Route 1 (50.1%) and Route 2 (49.9%)."`
- Confidence is **never** artificially inflated to $0.99$ when genuine routing alternatives exist.

---

## Multi-Observation Trajectory Chaining

For vehicle identities with $>2$ observations ($A \rightarrow B \rightarrow C \rightarrow D$):
- Decomposes observations into sequential transitions: $(A \rightarrow B)$, $(B \rightarrow C)$, $(C \rightarrow D)$.
- Reconstructs each segment independently against the road network.
- Concatenates edges and junction nodes into `complete_route_edges` and `complete_route_nodes`.
- Evaluates overall trajectory confidence using geometric mean of segment confidences.
- Handles stationary vehicle cases ($d = 0$, $A \rightarrow A$) cleanly without path search.

---

## Day 3 Output Contract Schema

```json
{
  "identity_id": "VEHICLE_CANDIDATE_002",
  "observations_count": 3,
  "cameras_visited": ["cam_01", "cam_02", "cam_03"],
  "start_timestamp": 12.0,
  "end_timestamp": 132.0,
  "total_time_seconds": 120.0,
  "total_distance_m": 909.3,
  "overall_confidence": 1.0,
  "is_ambiguous": false,
  "most_likely_route": ["road_01", "road_02"],
  "complete_route_nodes": ["junc_01", "junc_02", "junc_03"],
  "segments": [
    {
      "segment_id": "obs_002->obs_003",
      "identity_id": "VEHICLE_CANDIDATE_002",
      "start_observation": {
        "observation_id": "obs_002",
        "camera_id": "cam_01",
        "timestamp_seconds": 12.0,
        "node_id": "junc_01"
      },
      "end_observation": {
        "observation_id": "obs_003",
        "camera_id": "cam_02",
        "timestamp_seconds": 52.0,
        "node_id": "junc_02"
      },
      "time_difference_seconds": 40.0,
      "candidate_routes": [
        {
          "route_id": "route_01",
          "route": ["road_01"],
          "nodes": ["junc_01", "junc_02"],
          "distance_m": 414.8,
          "estimated_travel_time_s": 42.7,
          "min_travel_time_s": 29.9,
          "required_speed_kmh": 37.3,
          "speed_limit_kmh": 50.0,
          "feasible": true,
          "feasibility_status": "feasible",
          "estimated_likelihood": 1.0,
          "raw_score": 0.9922,
          "explanation": "Feasible route (414.8m): Required speed 37.3 km/h is compatible with road speed limit (50.0 km/h)."
        }
      ],
      "most_likely_route": ["road_01"],
      "confidence": 1.0,
      "is_ambiguous": false,
      "ambiguity_reason": null,
      "status": "success"
    }
  ]
}
```

---

## Edge-Case Handling & Robustness

The Day 3 inference engine was verified against 12 core edge cases:
- **CASE A (One Clear Route):** Single path correctly identified and scored.
- **CASE B (Multiple Feasible Routes):** Multiple alternative corridors preserved with normalized likelihoods.
- **CASE C (Shortest Route Infeasible):** Shortest urban path rejected due to low speed limit while longer expressway route is selected.
- **CASE D (All Routes Infeasible):** Extreme travel speeds correctly flag all candidates as `feasible: false`.
- **CASE E (Stationary Observations):** Same camera at different timestamps produces stationary trajectory ($d=0$) without crash.
- **CASE F (Reversed Timestamps):** $\Delta t < 0$ fails safely with clear explanatory reason.
- **CASE G (Unassociated Camera):** Out-of-bounds camera coordinates fail safely without route fabrication.
- **CASE H (No Path Exists):** Disconnected or one-way trapped nodes report `"no_path"`.
- **CASE I (Very Small $\Delta t$):** High required speed flagged physically impossible.
- **CASE J (Nearly Identical Alternatives):** Symmetric paths output $\sim 50\% / 50\%$ likelihoods with `is_ambiguous = true`.
- **CASE K (Multi-Observation Chaining):** $>2$ observations chain into smooth multi-segment paths.
- **CASE L (Missing Optional Fields):** None vehicle types or missing coordinates resolve gracefully.

---

# Day 3 Finalization & Member 3 Integration

### End-to-End Team Pipeline Workflow:
```text
┌───────────────────────────────┐
│           MEMBER 1            │
│  Perception & Spatial Graph   │
│  - Camera metadata            │
│  - Road network graph (OSM)   │
│  - Observation detections     │
└──────────────┬────────────────┘
               │ Observations + Road Graph
               ▼
┌───────────────────────────────┐
│     MEMBER 2 (THIS REPO)      │
│   Mobility Inference Engine   │
│  - Pairwise Identity Fusion   │
│  - Identity Graph Clustering  │
│  - Candidate Route Generation │
│  - Feasibility & Scoring      │
│  - NormalizedTrajectory Adapt │
└──────────────┬────────────────┘
               │ NormalizedTrajectory Payloads
               ▼
┌───────────────────────────────┐
│           MEMBER 3            │
│ Decision Intelligence & Flow  │
│  - Mobility Graph             │
│  - Trajectory → Road Flow     │
│  - BPR Congestion & HHI       │
│  - OD Flow Analysis           │
│  - Bottleneck & Simulation    │
└───────────────────────────────┘
```

---

## Member 3 Integration Contract (`NormalizedTrajectory`)

Defined in [schemas/normalized_trajectory_schema.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/schemas/normalized_trajectory_schema.py) and adapted via [inference/member3_adapter.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/member3_adapter.py):

```text
NormalizedTrajectory
├── track_id            (str: unique vehicle identity or track identifier)
├── origin_node         (str: starting road junction node ID)
├── destination_node    (str: ending road junction node ID)
├── vehicle_weight      (float: vehicle count / demand multiplier, default: 1.0)
├── candidate_routes[]
│   ├── nodes           (list[str]: ordered junction node IDs, length >= 2)
│   ├── probability     (float: normalized relative likelihood in [0.0, 1.0])
│   └── metadata        (dict: travel distance, speed limits, edge IDs)
└── time_window
    ├── start           (float | str: departure timestamp)
    └── end             (float | str: arrival timestamp)
```

### Downstream Demand Semantics:
Member 3 computes expected link traffic flow using:
$$\text{route demand} = \text{vehicle\_weight} \times \text{route\_probability}$$

### Decoupling Vehicle Weight vs Probability:
- **`vehicle_weight`**: Represents physical vehicle demand volume or Passenger Car Unit (PCU) equivalence (e.g. `1.0` for passenger cars, `2.5` for heavy commercial vehicles or buses).
- **`route_probability`**: Represents the normalized relative likelihood of route choice ($\sum_{i} P_i = 1.0$).
- **Clean Separation**: Weight is strictly a demand multiplier; probability is strictly a routing likelihood.

### Adapter Layer Functions:
- `adapt_trajectory_segment_to_normalized(segment, vehicle_weight=1.0)`: Converts pairwise Day 3 `TrajectorySegment` into `NormalizedTrajectory`.
- `adapt_vehicle_trajectory_to_normalized(trajectory, vehicle_weight=1.0)`: Converts multi-observation `VehicleTrajectory` (A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ D) into origin-to-destination corridor routes with joint probabilities.
- `adapt_trajectories_to_batch_payload(trajectories, default_weight=1.0)`: Generates batch JSON payload conforming to Member 3's flow aggregator.

---

## Shared Spatial Graph (`data/synthetic/city_network.json`)

Member 3's validated spatial graph is directly ingested as the shared spatial source:
- **14 Junctions (`J01` ... `J14`)**: North Gate Terminal, North Junction, Midtown Circle, Central Square, South Hub Terminal, etc.
- **28 Directed Road Segments (`R01` ... `R28`)**: Expressways, arterials, bypasses, and urban connectors.
- **Road Properties**: `from_node`, `to_node`, `distance_km` (automatically converted to `distance_m = distance_km * 1000.0`), `speed_limit_kmph`, `capacity_vph`, and `is_closed`.
- **Closed Road Handling**: Closed segments (`is_closed: true`) are excluded from routing adjacency, matching Member 3's `MobilityGraph`.

---

# Day 4: Sparse / Missing-Camera Trajectory Inference (Day 4 Complete)

Reasoning about vehicle movement across unobserved intervals (gaps) between sightings of the same cross-camera vehicle identity without synthesizing fake observation records.

### Core Architecture & Guiding Principles:
1. **Zero Observation Fabrication**: If cameras along an intermediate corridor did not observe the vehicle, the system **never** creates synthetic observation records claiming they did.
2. **Explicit Gap Representation**: An unobserved interval is represented as:
   - Observed endpoints (Start Observation $A$ and Later Observation $B$)
   - Unobserved interval duration ($\Delta t = T_B - T_A$)
   - Set of candidate hidden routes through the road network
   - Road-aware temporal feasibility & relative estimated likelihoods
   - Uncertainty & ambiguity surfacing
3. **Reused Day-3 Spatial Machinery**: Directly traverses Member 3's shared graph (`data/synthetic/city_network.json`), strictly respecting directed road constraints, one-way traps, closed roads (`is_closed: true`), distance, and speed limits.
4. **Member 3 Backward Compatibility**: Adapted directly into `NormalizedTrajectory` with Day-4 metadata (`gap_detected`, `gap_duration_seconds`, `observed_endpoints`, `inferred_segment`, `inference_reason`, `gap_state`, `observations_used`).

### Deliverables & Modules:
- **Gap Schema ([schemas/gap_schema.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/schemas/gap_schema.py)):** `SparseObservationGap` dataclass.
- **Sparse Inference Engine ([inference/sparse_engine.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/sparse_engine.py)):**
  - `detect_observation_gaps(observations, road_graph)`: Classifies intervals into direct single-hop, stationary, or unobserved gap.
  - `infer_sparse_gap(obs_a, obs_b, road_graph, ...)`: Infers candidate hidden routes, calculates required speeds, rejects impossible corridors, normalizes relative likelihoods, and detects routing ambiguity.
  - `infer_sparse_identity_trajectory(identity_data, road_graph, ...)`: Evaluates multi-observation vehicle journeys with mixed direct and unobserved intervals.
- **Member 3 Adapter Extension ([inference/member3_adapter.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/inference/member3_adapter.py)):**
  - `adapt_sparse_gap_to_normalized(gap, vehicle_weight=1.0)`: Converts a `SparseObservationGap` into Member 3's `NormalizedTrajectory`.
- **Day 4 Test Scenarios Fixture ([data/synthetic/day4_sparse_scenarios.json](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/data/synthetic/day4_sparse_scenarios.json)):** Controlled synthetic test fixture covering Cases 1 through 10.
- **Day 4 Test Suite ([tests/test_day4_sparse_inference.py](file:///Users/yanalavivekreddy/.gemini/antigravity-ide/scratch/urbantrack-ai/tests/test_day4_sparse_inference.py)):** 15 focused tests validating gap detection, Cases 1–10, zero observation fabrication, and Member-3 contract compatibility.

---

# Day 5: Camera Reliability & Uncertainty Propagation (Day 5 Complete)

Principled framework for tracking sensor trust, detection quality, and uncertainty without conflating sensor quality with behavioral deviation or probability of guilt.

### Core Architecture & Semantic Separations:
1. **Camera Reliability ($R_{cam} \in [0.1, 1.0]$)**: Historical sensor performance and environmental conditions.
2. **Observation Reliability**: Sensor-level trust combining camera reliability, detection confidence, and physical verification.
3. **Identity Match Likelihood**: Spatio-temporal and visual feature compatibility across cameras.
4. **Trajectory Reliability & Uncertainty**: Explicit separation of measurement trust from candidate route ambiguity.

---

# Day 6: City Mobility Graph & Traffic Flow (Day 6 Complete)

Aggregates individual vehicle trajectories into city-scale network flow dynamics and macro mobility metrics.

### Core Capabilities:
1. **Flow Conservation & Normalization**: Fractional route allocation conserving total vehicle weight ($\sum P(r) = 1.0$).
2. **Time-Windowed Demand**: Normalized hourly demand rate ($\text{vph}$) based on explicit time-window durations.
3. **Road Capacity & Utilization**: Empirical volume-to-capacity metrics evaluated against physical road design specifications.
4. **Network Centrality & Bottlenecks**: Betweenness centrality combined with utilization to flag macro network bottlenecks.

---

# Day 7: City-Scale Anomaly Detection & Investigation (Day 7 Complete & Locked)

Multi-dimensional anomaly detection and operational investigation reasoning over reconstructed trajectories, candidate route hypotheses, road utilization, and network topology.

### Core Operating Principle:
> **"An anomaly indicates deviation from a configured behavioral, physical, or network baseline. It does not establish intent, wrongdoing, or causality."**

### Explicit Conceptual Separations:
UrbanTrack AI strictly enforces that:
$$\text{INVALID DATA} \neq \text{PHYSICAL INCONSISTENCY} \neq \text{BEHAVIORAL ANOMALY} \neq \text{INVESTIGATION PRIORITY}$$

1. **Data Quality (`DATA_QUALITY`, `TEMPORAL_INCONSISTENCY`)**:
   - Malformed fields, missing required schemas, non-numeric timestamps, or inverted time intervals ($\Delta t < 0$) are categorized as `INVALID_INPUT` / `DataQualityStatus.INVALID`.
   - **Never** interpreted as a vehicle behaving anomalously; processing is safely halted with zero fabricated behavioral scores.
2. **Physical Inconsistency (`PHYSICAL_INCONSISTENCY`)**:
   - Travel speeds exceeding physical boundaries (e.g. $> 120\text{ km/h}$ for urban vehicles across candidate corridors).
   - Reflects physical impossibility rather than driver behavior.
3. **Network Constraint Inconsistency (`NETWORK_CONSTRAINT_INCONSISTENCY`)**:
   - An inferred route traversing a closed road segment is flagged as: *"The inferred route is incompatible with the current road-network state."*
   - **Never** implies suspicious intent or wrongdoing.
4. **Behavioral Anomaly (`BEHAVIORAL_ANOMALY`)**:
   - Substantial travel-time deviation from configured origin-destination baselines ($T_{meas} \gg T_{base}$ or $T_{meas} \ll T_{base}$).
   - Route corridor divergence from expected historical paths.
   - Evaluated **only** when a valid, comparable baseline exists. If no baseline is available, the status is explicitly set to `INSUFFICIENT_EVIDENCE`.
5. **Network Anomaly (`NETWORK_ANOMALY`)**:
   - Segments where expected demand substantially exceeds designed capacity ($utilization > 1.0$).
   - High betweenness centrality combined with high utilization is categorized as a **network bottleneck candidate**, distinct from vehicle behavior.
6. **Separation of Anomaly Score, Reliability, and Uncertainty**:
   - `overall_score \in [0, 1]` represents normalized deviation magnitude (NOT a probability of crime, guilt, or event occurrence).
   - Sensor reliability and route uncertainty are preserved independently to inform operational triage:
     - *High Anomaly + High Reliability* $\to$ Stronger investigation candidate (`HIGH_PRIORITY` / `INVESTIGATE`).
     - *High Anomaly + Low Reliability* $\to$ Requires verification before action (`WATCH`).
     - *Low Anomaly + High Reliability* $\to$ Confidently normal (`NORMAL`).
     - *Low Anomaly + Low Reliability* $\to$ Insufficient evidence (`INSUFFICIENT_EVIDENCE`).
7. **Preservation of Route Ambiguity**:
   - Competing candidate route hypotheses (e.g. probabilities $0.43, 0.31, 0.26$) remain explicitly ambiguous; top-route selection is never treated as certainty.
8. **Removal of Overclaimed Terminology**:
   - Pejorative or speculative labels ("loitering", "unexpected stop", "suspicious", "criminal", "stolen", "malicious") are banned from the codebase and reports. Replaced with neutral, factual descriptions ("substantial travel-time deviation", "travel time substantially exceeds configured baseline", "investigation candidate").

---

## Reproducing Demos & Tests

### 1. Run Complete Unit Test Suite (159 tests across Days 1–7)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Run Day 7 Anomaly Detection Test Suite (20 tests)
```bash
python3 -m unittest tests/test_day7_anomaly.py -v
```

### 3. Run Day 6 City Mobility Test Suite (17 tests)
```bash
python3 -m unittest tests/test_day6_mobility.py -v
```

### 4. Run Day 5 Reliability & Uncertainty Test Suite (17 tests)
```bash
python3 -m unittest tests/test_day5_reliability.py -v
```

### 5. Run Day 4 Sparse / Missing-Camera Test Suite (15 tests)
```bash
python3 -m unittest tests/test_day4_sparse_inference.py -v
```

### 6. Run Member 3 Integration Contract Tests (15 tests)
```bash
python3 -m unittest tests/test_member3_integration.py -v
```

### 7. Run Day 3 End-to-End Validation
```bash
python3 tests/validate_day3_end_to_end.py
```

### 8. Run Day 2 Synthetic Benchmark (9 Scenarios)
```bash
python3 run_benchmark.py
```

### 9. Run Real Perception Evaluation (Kanishka's Feed)
```bash
python3 run_real_data.py
```

---

## Technical Honesty, Semantics & Limitations

1. **Identity Scoring Semantics:** The identity engine uses **multimodal evidence-based identity scoring**. Edges in the identity graph are formed when pairwise evidence meets the configured decision threshold (`0.70` evidence score). These values are normalized relative evidence scores and are **not** statistically calibrated probabilities or Bayesian posteriors.
2. **Normalized Route Likelihoods, Not Probabilities:** Candidate route scores represent normalized relative estimated likelihoods among feasible corridors given network geometry and speed limits. The schema field `probability` is preserved for interface compatibility, but does not represent a calibrated statistical probability.
3. **Route-Distribution Entropy vs Reliability:** Route Shannon entropy measures dispersion/ambiguity across competing feasible routes. Sensor and observation reliability are heuristic estimates of evidence trustworthiness and are conceptually distinct from entropy.
4. **Physical Demand Conservation:** Road flow is modeled as $\text{vehicle\_weight } (W) \times \text{route\_allocation } (P_i)$. Reliability is **never** multiplied into physical traffic demand. Demand conservation is validated at the trajectory allocation level ($\sum P_i \approx 1.0$).
5. **Counterfactual Simulation:** Counterfactual modeling is hypothetical network scenario analysis (evaluating how currently modeled demand reallocates across surviving feasible corridors under an explicit intervention). It is **not** an exact future prediction or traffic forecast.
6. **Kanishka Perception Feed Compatibility:** In the evaluated Kanishka perception feed (2,503 records processed in this run), cross-camera vehicle re-identification appearance embeddings and license plates are unavailable. In this validation run, no fabricated identity links, routes, traffic demand, or anomalies were introduced from unavailable perception evidence; observations remained separately identified rather than forcefully merged.
7. **Production Readiness:** Hackathon/demo-ready; production deployment requires authoritative camera geolocation and GIS junction mapping, hardware shared time synchronization, metric camera calibration, production-validated road capacities, and privacy/governance infrastructure.

