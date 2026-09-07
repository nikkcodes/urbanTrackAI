# Member 3: Urban Mobility & Decision Intelligence Architecture (Phase 1)

## 1. Overview & System Role

UrbanTrackAI is a city-scale vehicle mobility intelligence system. In the overall conceptual pipeline:

```text
CCTV / ANPR cameras
        ↓
Perception (Member 1: detection, OCR, appearance embeddings)
        ↓
Identity Fusion & Trajectory Inference (Member 2: matching, candidate route probabilities)
        ↓
CITY MOBILITY GRAPH (Member 3: road network, flow aggregation, metrics)
        ↓
Traffic Analytics (OD analysis, bottlenecks, anomalies)
        ↓
Counterfactual Simulation (what-if road closures, rerouting)
        ↓
GIS Dashboard (real-time situational awareness)
```

**Member 3 Responsibility Boundary**:
- **Consumes**: Vehicle observations and probabilistic route candidates from Member 2.
- **Maintains**: The city road network representation, topology, free-flow and dynamic travel-time metrics, capacity constraints, and operational road states (active/closed).
- **Produces**: Aggregated road utilization, corridor delays, bottleneck warnings, counterfactual impact predictions, and geospatial data layers.
- **Strict Boundary**: Member 3 does **not** implement ANPR, OCR, vehicle detection, appearance Re-ID, camera reliability, or vehicle identity matching.

---

## 2. Graph & Data Models

### 2.1 Graph Formulation
The city is modeled as a directed, weighted multigraph/digraph:
$$G = (V, E)$$
where:
- $V$: Set of junctions, intersections, expressway interchanges, and sensor locations.
- $E$: Set of directed road segments $(u, v)$ with physical and operational attributes.

### 2.2 Node Model (`backend.mobility.models.Node`)
- `node_id` (`str`): Unique identifier (e.g. `J01`). Non-empty and sanitized.
- `name` (`Optional[str]`): Human-readable landmark or intersection name.
- `lat`, `lon` (`Optional[float]`): WGS-84 geographical coordinates (validated: $-90 \le \text{lat} \le 90$, $-180 \le \text{lon} \le 180$) for future GIS and map rendering.
- `metadata` (`Dict[str, Any]`): Extensible storage for zone classification, sensor IDs, and camera counts.

### 2.3 RoadSegment Model (`backend.mobility.models.RoadSegment`)
- `road_id` (`str`): Unique identifier (e.g. `R01`).
- `from_node` (`str`), `to_node` (`str`): Directed origin and destination nodes. Self-loops are strictly prohibited.
- `distance_km` (`float`): Physical road segment length in kilometers ($> 0$).
- `speed_limit_kmph` (`float`): Speed limit in km/h ($> 0$, prevents division by zero).
- `capacity_vph` (`float`): Practical vehicle throughput capacity in vehicles per hour ($> 0$).
- `free_flow_time_min` (`float`): Free-flow travel time in minutes:
  $$\text{free\_flow\_time\_min} = \frac{\text{distance\_km}}{\text{speed\_limit\_kmph}} \times 60$$
- `is_closed` (`bool`): Operational status flag.
- `metadata` (`Dict[str, Any]`): Extensible storage for road classification (expressway, arterial, urban), lane counts, and future live telemetry.

---

## 3. Road Closure & Counterfactual Design

A central requirement for urban decision intelligence is simulating road closures (incidents, construction, emergency cordons) without mutating or discarding the foundational road definition.

- **`close_road(road_id)`**:
  1. Marks `RoadSegment.is_closed = True`.
  2. Removes the directed edge from the active NetworkX routing graph.
  3. Preserves the road in the internal registry and retains all physical metadata.
- **`restore_road(road_id)`**:
  1. Marks `RoadSegment.is_closed = False`.
  2. Re-inserts the edge into the active NetworkX graph with original weights and attributes.
- **`copy(deep=True)`**:
  Produces an isolated replica of the entire `MobilityGraph`. This allows counterfactual simulation algorithms in future phases to run scenario evaluations (e.g. "What happens to downtown congestion if bridge R14 closes?") without affecting the live operational graph.

---

## 4. Route Discovery & Uncertainty Principle

In real-world urban networks, vehicles do not strictly adhere to the single shortest path. Member 2's trajectory inference pipeline generates multiple candidate routes with associated posterior probabilities:
$$P(\text{Route}_A) = 0.74, \quad P(\text{Route}_B) = 0.20, \quad P(\text{Route}_C) = 0.06$$

### 4.1 Route Discovery API (`backend.mobility.routes.get_candidate_routes`)
The routing engine discovers up to $k$ feasible, simple (loop-free) alternative paths ordered by travel time or distance using NetworkX shortest simple path algorithms:
```python
routes = get_candidate_routes(graph, source="J01", destination="J12", max_routes=4)
for r in routes:
    print(r.path, r.distance_km, r.free_flow_time_min)
```
- Closed road segments are automatically bypassed.
- Returns structured `RouteInfo` objects including node sequence, road segment IDs, cumulative distance (km), and free-flow travel time (minutes).

---

## 5. Technology Choices for Phase 1

1. **Python 3.14 Standard Library**:
   - Python dataclasses with `__post_init__` are utilized for `Node` and `RoadSegment` to guarantee strict runtime typing, validation, zero serialization overhead, and flawless compatibility with Python 3.14.
   - Built-in `unittest` enables instantaneous test discovery without external test runner dependencies, while remaining fully runnable under `pytest`.
2. **NetworkX (3.6.1)**:
   - High-maturity, pure-Python graph engine ideal for algorithmic graph manipulation, shortest simple path generation, and topology isolation.
   - Allows clean decoupling from database engines during early prototyping.

---

## 6. What is Intentionally Deferred to Future Phases

To maintain a clean, lightweight, and extensible foundation, the following components are **not** implemented in Phase 1:
- **Origin-Destination (OD) Matrix Analysis**: Multi-interval matrix construction from fused vehicle trajectories.
- **Probabilistic Flow Aggregation**: BPR (Bureau of Public Roads) congestion functions and volume-delay curves aggregating Member 2's route probability distributions.
- **Bottleneck & Anomaly Detection**: Graph-level betweenness centrality shifts and spillback detection.
- **PostgreSQL / PostGIS**: Spatial persistence layer and geospatial indexing.
- **Streaming Pipeline**: Apache Kafka / Redis pub-sub integration for high-velocity ANPR observation streams.
- **GIS Dashboard**: React / Mapbox / Leaflet frontend visualization.
