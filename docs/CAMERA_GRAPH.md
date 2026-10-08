# UrbanTrack AI - Camera Graph Topology Specification

This document defines the architecture, semantics, mathematical foundation, and empirical validation protocols for [`camera_graph.json`](file:///c:/UrbanTrack/data/config/camera_graph.json).

---

## 1. What `camera_graph.json` Represents

[`camera_graph.json`](file:///c:/UrbanTrack/data/config/camera_graph.json) models the directed transition topology of the multi-camera surveillance network across all 65 scenario-qualified video streams in the AI City Challenge 2022 Track 1 MTMC dataset.

- **Nodes (65)**: Represent individual video streams qualified by scenario (`CAM_<SCENARIO>_<CAM>`). Each node maps 1:1 to a physical camera record in [`camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json).
- **Edges (170)**: Represent directed physical connectivity or empirically demonstrated vehicle transitions between adjacent or corridor-linked cameras within the same operational scenario.
- **Attributes per edge**:
  - `source` and `target`: Canonical scenario-qualified camera IDs.
  - `relationship`: Fixed semantic string `"possible_transition"`.
  - `distance_m`: Great-circle Haversine distance in meters computed between camera coordinates.
  - `bearing_deg`: Initial forward compass bearing from source to target in degrees $[0.0, 360.0)$ ($0^\circ = \text{North}, 90^\circ = \text{East}$).
  - `evidence`: Authoritative visual map sources (e.g., `cam_loc/S0345.png`) and/or empirical trajectory ground truth (`validation/S05/gt/gt.txt`).
  - `confidence`: Confidence score (`"HIGH"`, `"MEDIUM"`, `"LOW"`).

---

## 2. Meaning of `"possible_transition"`

The edge property `"relationship": "possible_transition"` signifies:

> **A spatial and topological possibility of vehicle movement supported by physical roadway layout, adjacent intersection geometry, and/or observed dataset trajectories.**

It is strictly **NOT** an assertion that:
- Every vehicle moving past the source camera must transit to the target camera.
- All vehicle transitions between these two cameras occur without turnoffs or stops.
- Physical movements outside the graph are physically impossible in the real world.

Even when direct ground-truth evidence demonstrates vehicle transitions, edges preserve `"possible_transition"` semantics rather than being redefined as `"observed_transition"`. This maintains a consistent mathematical graph schema across all operational scenarios.

---

## 3. Difference Between Spatial Topology and Observed Transitions

A principled multi-camera tracking system must distinguish between two levels of information:

| Level | Definition | Source of Truth | Graph Role |
| :--- | :--- | :--- | :--- |
| **Spatial / Road Topology** | Physical connectivity of Dubuque roadways (two-way arterials, intersections, ramps). | Georeferenced maps (`cam_loc/*.png`), OpenStreetMap, homography ground planes. | Defines candidate travel corridors and road distance bounds. |
| **Observed Dataset Transitions** | Specific vehicle trips recorded during the limited evaluation capture windows (typically 1–2 hours). | MTMC ground truth (`gt.txt`), multi-camera bounding-box trajectories. | Empirically confirms traffic movements along candidate corridors. |

In AI City Challenge scenarios, video recordings may omit intermediate cameras (e.g., S05 records a subset of the University Avenue cameras used in S04). Ground-truth observations reflect vehicle trajectories through this specific camera subset, while spatial topology ensures the physical continuity of the roadway corridor remains modeled.

---

## 4. Why the Graph is a Soft Topological Prior (Not a Hard Filter)

Under no circumstances should cross-camera tracking logic implement hard pruning:
```python
# INCORRECT (Hard Rejection Filter):
if not graph.has_edge(cam_a, cam_b):
    reject_candidate(track_a, track_b)
```

The camera graph serves strictly as a **soft Bayesian prior**:
- **When an edge exists**: The spatial prior probability $P(\text{cam}_b \mid \text{cam}_a, \Delta t, \vec{v})$ is elevated based on travel distance, heading alignment, and corridor continuity.
- **When an edge does not exist**: The candidate is **not** discarded. Association remains governed by multi-modal evidence:
  1. OSNet Re-ID embedding cosine similarity (512-D)
  2. OCR / ANPR license plate alphanumeric matches
  3. Temporal feasibility ($\Delta t = t_b - t_a > 0$ and velocity $v = d / \Delta t \le v_{\max}$)
  4. Homography & GIS coordinate projection consistency
  5. Direction of travel and heading consistency

This soft prior architecture protects the tracking system against real-world anomalies such as U-turns, off-street parking, sensor blind spots, or camera occlusion.

---

## 5. How Scenario S05 Topology Was Validated

Scenario S05 records 19 cameras in Dubuque, Iowa, spanning the Hill Street and University Avenue corridors. Due to omitted bridge cameras from earlier scenarios (such as C011, C030–C032, and C038), the initial graph left `CAM_S05_C010` and `CAM_S05_C033` isolated (degree 0).

A full audit cross-referencing [`camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json) and S05 ground-truth trajectories (`validation/S05/*/gt/gt.txt`) resolved these isolated nodes by verifying six directed edges:

| Source | Target | Distance (m) | Forward Bearing | Direct GT Trips | Multi-hop GT Trips | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CAM_S05_C010` | `CAM_S05_C017` | 362.8 | 1.3° | 14 | 14 | **VERIFIED** |
| `CAM_S05_C017` | `CAM_S05_C010` | 362.8 | 181.3° | 15 | 15 | **VERIFIED** |
| `CAM_S05_C033` | `CAM_S05_C034` | 135.2 | 285.8° | 146 | 156 | **VERIFIED** |
| `CAM_S05_C034` | `CAM_S05_C033` | 135.2 | 105.8° | 10 | 10 | **VERIFIED** |
| `CAM_S05_C029` | `CAM_S05_C034` | 253.7 | 290.0° | 0 | 47 | **VERIFIED** |
| `CAM_S05_C034` | `CAM_S05_C029` | 253.7 | 110.0° | 48 | 50 | **VERIFIED** |

### Empirical Insights
1. **`CAM_S05_C010 <-> CAM_S05_C017`**: Direct North-South link connecting Hill St & W 5th St to University Ave at Loras Blvd. Ground truth records 14 northbound and 15 southbound vehicle transitions.
2. **`CAM_S05_C033 <-> CAM_S05_C034`**: Direct link connecting Asbury Rd & Cherry St to University Ave & Kirk St. Ground truth records 146 westbound and 10 eastbound transitions.
3. **`CAM_S05_C029 <-> CAM_S05_C034`**: Continuous two-way corridor along University Avenue across the unrecorded C030–C032 gap. Ground truth records 48 eastbound vehicles moving directly from C034 to C029. In the westbound direction, 47 vehicles traverse the corridor from C029 to C034 via the adjacent C033 field of view. Both directions are preserved to reflect the physical two-way corridor.

---

## 6. How Future Edges Should Be Added

When adding new edges to `camera_graph.json`:
1. **Preserve Scenario Qualification**: Sources and targets must use canonical scenario-qualified IDs (`CAM_<SCENARIO>_<CAM>`). Never connect cameras across different scenarios.
2. **Verify Coordinates**: Compute geodesic distance and forward bearing directly from [`camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json) using standard Haversine and forward azimuth equations.
3. **Audit Ground Truth**: Run [`scripts/validate_s05_gt_transitions.py`](file:///c:/UrbanTrack/scripts/validate_s05_gt_transitions.py) (or scenario-equivalent) to inspect vehicle trajectories.
4. **Never Force Component Collapse**: Do not add arbitrary edges solely to create a single connected component. Connectivity must reflect actual road topology and camera coverage.
5. **Run Validation**: Execute [`scripts/validate_camera_locations.py`](file:///c:/UrbanTrack/scripts/validate_camera_locations.py) and [`tests/test_camera_graph.py`](file:///c:/UrbanTrack/tests/test_camera_graph.py) before committing any changes.

---

## 7. Why Scenario-Qualified IDs Must Be Preserved

In the AI City Challenge dataset:
- Scenarios S01–S06 were recorded at different dates, times of day, and weather conditions.
- Numerical camera IDs (e.g., `C010`) are reused across scenarios (`CAM_S03_C010` vs. `CAM_S05_C010`).
- While physical poles may be shared, the video streams represent distinct temporal epochs with different operational contexts.
- Collapsing cameras across scenarios breaks tracking state machines, corrupts temporal window constraints ($\Delta t$), and invalidates multi-camera association matrices.

Canonical scenario-qualified identifiers (`CAM_S01_C001`, `CAM_S05_C010`, etc.) ensure complete referential clarity throughout the entire perception, tracking, and GIS visualization pipeline.
