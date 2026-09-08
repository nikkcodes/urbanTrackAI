# Member 3 — Phase 2: Probabilistic Trajectory → Expected Road Flow Aggregation

## 1. Phase 2 Purpose

The objective of Phase 2 is to establish an architectural bridge between upstream trajectory inference (Member 2) and downstream traffic analytics (Member 3 Phase 3 and Phase 4). 

Upstream vehicle tracking inherently possesses positional and route uncertainty when traversing urban sensor networks. Phase 2 converts multi-hypothesis, probability-weighted vehicle candidate routes into deterministic, segment-level expected traffic flows across the directed `MobilityGraph`.

```
                    ┌────────────────────────────┐
                    │ Upstream Member 2 Pipeline │
                    │ (Multi-camera Re-ID / OCR) │
                    └─────────────┬──────────────┘
                                  │ (Raw inference payloads)
                                  ▼
                    ┌────────────────────────────┐
                    │ Trajectory Adapter Layer   │
                    │  (Base / Mock / Future M2) │
                    └─────────────┬──────────────┘
                                  │ NormalizedTrajectory objects
                                  ▼
                    ┌────────────────────────────┐
                    │ Normalized Trajectory Core │
                    │ (CandidateRoute, Probs, OD)│
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ ExpectedFlowAggregator     │
                    │ (Graph mapping, Weighting) │
                    └─────────────┬──────────────┘
                                  │ FlowAggregationResult
                                  ▼
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌────────────────────────────┐              ┌────────────────────────────┐
│ Member 3 — Phase 3         │              │ Member 3 — Phase 4         │
│ - Traffic Volume (V)       │              │ - Origin-Destination (OD)  │
│ - Capacity Utilization     │              │ - Corridor Analysis        │
│ - BPR Travel Times         │              │ - Bottleneck Identification│
│ - Congestion Metrics       │              │ - Counterfactual Closures  │
└────────────────────────────┘              └────────────────────────────┘
```

---

## 2. Input Boundary

The input layer is completely decoupled from Member 2's internal camera processing, bounding-box associations, Re-ID embeddings, and graph search heuristics.

The aggregation engine consumes **only** instances of `NormalizedTrajectory`. External payloads enter through dedicated adapters (`BaseTrajectoryAdapter` interface) rather than being ingested directly by the flow calculation core.

---

## 3. Normalized Trajectory Concept

The internal data model in `backend/flow/models.py` captures only what downstream analytics genuinely requires:

### `CandidateRoute`
- `nodes: List[str]`: An ordered sequence of junction node IDs representing a continuous physical path across the graph (length $\ge 2$).
- `probability: float`: Float in $[0.0, 1.0]$ representing the conditional likelihood of this path.
- `metadata: Dict[str, Any]`: Confidence scores, route descriptions, or sensor sequence notes.

### `NormalizedTrajectory`
- `track_id: str`: Unique identifier for the vehicle track or trip.
- `origin_node: str`: Entry junction ID.
- `destination_node: str`: Exit junction ID.
- `candidate_routes: List[CandidateRoute]`: Non-empty collection of candidate routes.
- `vehicle_weight: float`: Count or statistical weighting of vehicles (default: `1.0`, strictly $> 0$).
- `timestamp: Optional[str]`: Observation or trip completion time.
- `time_window_start: Optional[str]`: Start of analysis interval (e.g. `"08:00:00"`).
- `time_window_end: Optional[str]`: End of analysis interval (e.g. `"08:15:00"`).
- `metadata: Dict[str, Any]`: Vehicle classification, sensor count, etc.

---

## 4. Mock Adapter

The `MockTrajectoryAdapter` in `backend/flow/adapters.py` reads mock trajectory payloads from:
- JSON file paths (`data/synthetic/mock_trajectories.json`)
- In-memory dictionaries containing `{"trajectories": [...]}`
- Raw lists of trajectory dictionaries
- Single trajectory items

It flexibly normalizes alternative dictionary keys (such as `"path"` vs `"nodes"` or `"weight"` / `"count"` vs `"vehicle_weight"`) into strongly-typed `NormalizedTrajectory` instances.

The mock fixture (`data/synthetic/mock_trajectories.json`) is explicitly designated with clear metadata disclaimers as temporary synthetic validation data.

---

## 5. Expected-Flow Calculation

For each probabilistic trajectory, its vehicle weight is distributed across its candidate routes proportional to their probability. Crucially, uncertainty is preserved—trajectories are **never** collapsed to their single highest-probability route:

$$\text{flow\_contribution}(r, e) = w_i \times p_{i, r} \quad \forall e \in \text{roads}(r)$$

$$\text{expected\_flow}(e) = \sum_{i \in \text{trajectories}} \sum_{r \in \text{routes}(i)} w_i \cdot p_{i, r} \cdot \mathbb{I}(e \in r)$$

Where:
- $w_i$ is the `vehicle_weight` of trajectory $i$.
- $p_{i, r}$ is the probability of candidate route $r$ in trajectory $i$.
- $\mathbb{I}(e \in r)$ indicates whether road segment $e$ is traversed by candidate route $r$.

All road flow results are rounded to 6 decimal places for floating-point determinism and stored in `RoadFlow` objects.

---

## 6. Node-Path → Road-ID Mapping

Phase 1 route discovery represents candidate routes as ordered node sequences:
$$[\text{J01}, \text{J02}, \text{J05}]$$

The flow aggregation engine maps consecutive node pairs `(u, v)` directly to directed road segments registered in `MobilityGraph`:
$$\text{J01} \to \text{J02} \implies \text{R01}$$
$$\text{J02} \to \text{J05} \implies \text{R08}$$

Road IDs are retrieved dynamically via `graph.get_road_by_nodes(u, v)`. Road IDs are **never hardcoded**. If any consecutive node pair does not exist as an edge in the network, an `InvalidRouteError` is raised immediately.

---

## 7. Probability Validation

Each candidate route collection must form a mathematically sound discrete probability distribution. The following rules are enforced during trajectory instantiation and aggregation validation:
1. Every candidate route probability must be a finite numerical value.
2. Every candidate route probability must satisfy $0.0 \le p \le 1.0$.
3. The sum of candidate route probabilities must equal $1.0$ within a configurable numerical tolerance:
   $$\left|\sum_{r} p_r - 1.0\right| \le \text{tolerance} \quad (\text{default: } 10^{-6})$$

> **Strict Non-Silent Failures**: The system **never** silently renormalizes incorrect probabilities (e.g. probabilities summing to $0.80$ or $1.40$). An explicit `ProbabilityValidationError` is raised. Silently altering invalid distributions would mask upstream Member 2 tracking errors and corrupt downstream traffic metrics.

---

## 8. Vehicle Weighting

Trajectories support an optional `vehicle_weight` attribute (default: `1.0`):
- Models single passenger vehicles ($w = 1.0$).
- Models multi-vehicle batches, platoons, or commercial convoys ($w > 1.0$, e.g. $w = 2.0$).
- Models confidence-adjusted or subsampled trips.

Expected road flow scales linearly with vehicle weight:
$$\Delta \text{flow}(e) = \text{vehicle\_weight} \times \text{probability}$$

Vehicle weights $\le 0.0$ or non-finite numbers are strictly rejected.

---

## 9. Time-Window Handling

To allow Phase 3 to evaluate temporal traffic variations and peak periods:
- `NormalizedTrajectory` stores optional `time_window_start`, `time_window_end`, and `timestamp` fields.
- `ExpectedFlowAggregator.aggregate()` accepts an optional `time_window` tuple and attaches it to the resulting `FlowAggregationResult` and each `RoadFlow`. If all trajectories in a batch share identical window boundaries, they are automatically propagated.
- `ExpectedFlowAggregator.aggregate_by_time_window()` automatically partitions a mixed stream of trajectories by `(start, end)` time intervals and returns a dictionary of isolated `FlowAggregationResult` objects for each discrete window.

If time window fields are omitted, the aggregator functions normally without error.

---

## 10. Duplicate-Road Behavior

If an anomalous or looped candidate route traverses the same directed road segment more than once (for instance: $\text{J01} \to \text{J02} \to \text{J01} \to \text{J02} \to \text{J05}$ where road $\text{R01}$ is traversed twice):

> **Documented Traversal Rule**:
> Each road segment is counted **once per trajectory candidate route** for expected vehicle flow.
> $$\Delta \text{flow}(\text{R01}) = w_i \times p_{i, r} \quad (\text{not } 2 \cdot w_i \cdot p_{i, r})$$

**Rationale**: We are modeling the expected count of unique vehicles utilizing a road segment from a probabilistic trip, rather than cumulative lap counts. Deduplication is performed by preserving unique road IDs per candidate route:
```python
unique_route_roads = list(dict.fromkeys(route_road_ids))
```

---

## 11. Member 2 Integration Strategy

Member 2 is currently completing multi-camera tracking, vehicle Re-ID, and trajectory inference. Our integration architecture guarantees zero lock-in:

1. **No Shared Dependency on M2 Code**: `backend/flow/` contains zero imports from Member 2 modules.
2. **Pluggable Adapter Pattern**: When Member 2 finalizes their output format (whether dictionary, JSONL, or Protobuf), Member 3 will write a thin adapter subclassing `BaseTrajectoryAdapter`:
   ```python
   class Member2TrajectoryAdapter(BaseTrajectoryAdapter):
       def adapt(self, m2_payload) -> List[NormalizedTrajectory]:
           ...
   ```
3. **Preserved Downstream Stability**: Once converted to `NormalizedTrajectory`, all downstream flow aggregation, Phase 3 metrics, and Phase 4 OD matrices run without any code modification.

---

## 12. How Phase 3 Consumes Output

Phase 3 will import `FlowAggregationResult` and `RoadFlow` from `backend.flow`. Phase 3 will consume:
- `rf.road_id`: Road segment key.
- `rf.expected_flow`: Expected vehicle volume $V$ over time window.
- `rf.time_window_start` / `rf.time_window_end`: Analysis interval.

Using the Phase 1 `MobilityGraph` (`road.capacity_vph`, `road.free_flow_time_min`), Phase 3 will compute:
- Volume-to-Capacity ratio: $\text{V/C} = \frac{V}{C}$
- Congested travel time via Bureau of Public Roads (BPR) function:
  $$t_{\text{congested}} = t_0 \left(1 + \alpha \left(\frac{V}{C}\right)^\beta\right)$$
- Congestion indices, bottleneck indicators, and level of service (LOS).

---

## 13. How Phase 4 Accesses Trajectory-Level Information

While `RoadFlow` aggregates traffic at the segment level, `FlowAggregationResult` explicitly retains the full collection of `NormalizedTrajectory` instances in its `trajectories` attribute:
```python
result = aggregator.aggregate(trajectories)
raw_trips = result.trajectories
```

Phase 4 can directly access:
- `t.origin_node` and `t.destination_node` to build Origin-Destination (OD) trip matrices:
  $$T_{od} = \sum_{t \in \text{trajectories}} t.\text{vehicle\_weight} \cdot \mathbb{I}(O(t)=o \land D(t)=d)$$
- `t.candidate_routes` to perform major corridor identification and route choice diversity analysis.
- Network sensitivity / counterfactual closure analysis by running `aggregator.aggregate()` against a modified `MobilityGraph.copy()`.

---

## 14. What is Intentionally NOT Implemented Yet

To adhere to strict modular milestones:
- **No BPR travel time equations**: Belongs in Phase 3.
- **No congestion / V/C scoring**: Belongs in Phase 3.
- **No speed-flow curves**: Belongs in Phase 3.
- **No full OD matrices or corridor clustering**: Belongs in Phase 4.
- **No REST/FastAPI endpoints**: Belongs in the API phase in `backend/api/`.
- **No frontend dashboards**: Belongs in the UI visualization phase.
- **No streaming message brokers** (Kafka/RabbitMQ/Redis): Unnecessary for batch/micro-batch analytics at this stage.
