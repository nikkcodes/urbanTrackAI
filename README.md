# UrbanTrack AI - Identity Fusion Engine (Day 2 Complete)

City-scale multi-camera vehicle trajectory inference system for **UrbanTrack AI**.

**Role**: Vivek — Mobility Inference Engineer (Member 2)  
**Scope**: Cross-camera vehicle identity fusion, pairwise feasibility scoring, identity graph representation, and candidate identity clustering.

---

## Objective

Answer the core question:
> *"Given two vehicle observations from different cameras, how likely is it that they represent the same physical vehicle?"*

---

## System Architecture & Pipeline Flow

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

## Input Schemas

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
- Handles stationary/loitering cases ($d = 0$, $A \rightarrow A$) cleanly without path search.

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

## Reproducing Demos & Tests

### 1. Run Complete Unit Test Suite (58 tests)
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 2. Run Day 3 End-to-End Validation
```bash
python3 tests/validate_day3_end_to_end.py
```

### 3. Run Day 2 Synthetic Benchmark
```bash
python3 run_benchmark.py
```

### 4. Run Real Perception Evaluation (Kanishka's Feed)
```bash
python3 run_real_data.py
```

---

## Technical Honesty & Limitations

1. **Uncalibrated Relative Likelihoods:** Candidate route scores are normalized relative likelihoods based on travel speed and path distance. They are **not** Bayesian posterior probabilities.
2. **Synthetic Road Graph:** Because Nikhilesh's full city-scale road network is not yet checked in, Day 3 uses a deterministic, geometrically grounded synthetic road network in Hyderabad coordinate space (`data/roads/synthetic_road_graph.json`). It is not claimed to be live government GIS data.
3. **Kanishka Perception Feed Compatibility:** The current perception feed lacks persistent Re-ID embeddings, resulting in singletons from Day 2. Day 3 handles singletons safely ($d=0$, confidence=1.0) without fabricating artificial multi-camera trajectories.

