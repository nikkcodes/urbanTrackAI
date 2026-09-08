# Member 3 — Phase 3: Traffic Metrics & Intelligence Engine

## 1. Phase 3 Purpose

The objective of Phase 3 is to translate expected vehicle flows on network road segments (from Phase 2) into operational traffic intelligence:
- Quantifying road capacity utilization ($V/C$).
- Computing dynamic congested travel times using the Bureau of Public Roads (BPR) volume-delay formulation.
- Deriving continuous, deterministic congestion scores.
- Classifying congestion states into standard traffic operational categories (`FREE`, `MODERATE`, `HEAVY`, `SEVERE`).
- Aggregating network-wide summaries with simple and flow-weighted statistics.
- Evaluating time-window traffic variations across temporal slices.

```
                    ┌────────────────────────────┐
                    │ Member 3 — Phase 2         │
                    │ Expected Road Flow Output  │
                    │ (RoadFlow, FlowAggregation)│
                    └─────────────┬──────────────┘
                                  │ RoadFlow records
                                  ▼
                    ┌────────────────────────────┐
                    │ TrafficMetricsCalculator   │
                    │ (BPR, Capacity, Delay)     │
                    └─────────────┬──────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
    ┌────────────────────────────┐  ┌────────────────────────────┐
    │ TrafficMetric Records      │  │ NetworkTrafficSummary      │
    │ (Utilization, t_est,       │  │ (Total Flow, Level Counts, │
    │  Congestion Score & Level) │  │  Flow-Weighted Averages)   │
    └─────────────┬──────────────┘  └─────────────┬──────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Downstream Phases          │
                    │ Phase 4: OD & Bottlenecks  │
                    │ Phase 5: Anomaly Detection │
                    │ Phase 6: Counterfactuals   │
                    └────────────────────────────┘
```

---

## 2. Input from Phase 2

Phase 3 operates directly on Phase 2's output without altering or duplicating flow aggregation:
- Ingests `FlowAggregationResult` or individual `RoadFlow` records (`road_id`, `expected_flow`, `from_node`, `to_node`, `time_window_start`, `time_window_end`).
- Combines `RoadFlow` records with physical infrastructure attributes from the Phase 1 `MobilityGraph` (`capacity_vph`, `free_flow_time_min`, `distance_km`, `speed_limit_kmph`).

Phase 3 treats `expected_flow` as a continuous expected vehicle-segment traversal count accumulated during the Phase 2 aggregation window. It is not vehicles/hour and is not a unique city vehicle count: one trajectory can contribute to multiple road segments. It does not collapse or discretize flow into single hard routes.

---

## 3. Traffic Metric Definitions

For each evaluated road segment, `TrafficMetric` provides:

| Metric | Symbol / Type | Unit | Description |
|---|---|---|---|
| `expected_flow` | $V_w$ | vehicle-segment traversals | Aggregated expected road traversals during the Phase 2 window. |
| `hourly_flow` | $V$ | vehicles/hour | `expected_flow` normalized by the validated window duration. |
| `capacity_vph` | $C$ | vehicles/hour | Segment physical design capacity. |
| `utilization_ratio` | $V/C$ | dimensionless | Volume-to-capacity utilization ratio (unclamped). |
| `free_flow_time_minutes` | $t_0$ | minutes | Baseline free-flow travel time at speed limit. |
| `estimated_travel_time_minutes` | $t$ | minutes | Congested travel time derived from the BPR model. |
| `congestion_score` | $S$ | $[0.0, 1.0)$ | Normalized delay fraction $(t - t_0) / t$. |
| `congestion_level` | `CongestionLevel` | enum | State: `FREE`, `MODERATE`, `HEAVY`, `SEVERE`. |
| `time_window_start` / `end` | strings | time | Temporal slice boundaries. |

---

## 4. Capacity Utilization Formula

$$\text{hourly\_flow} = \frac{\text{expected\_flow}}{\text{window duration in hours}}$$

$$\text{utilization\_ratio} = \frac{\text{hourly\_flow}}{C}$$

Where:
- `expected_flow` is the accumulated expected vehicle-segment traversal count from Phase 2.
- `hourly_flow` is the window-normalized road rate in vehicles/hour.
- $C$ is `capacity_vph` (strictly positive finite number $> 0$).

### Design Decisions & Safety
- **Zero/Invalid Capacity**: If $C \le 0$ or non-finite, the calculator raises an explicit `ValueError`. Silent division by zero is strictly prohibited.
- **No Arbitrary Clamping**: When $V > C$, the ratio exceeds $1.0$ (e.g. $1.25$ or $1.50$) rather than pretending the segment is at $100\%$ capacity. Oversaturation is physically meaningful for bottleneck detection in Phase 4.

---

## 5. Bureau of Public Roads (BPR) Travel Time Model

$$t = t_0 \left(1 + \alpha \left(\frac{\text{hourly\_flow}}{C}\right)^\beta\right)$$

Where:
- $t_0$ = `free_flow_time_minutes` $= \frac{\text{distance\_km}}{\text{speed\_limit\_kmph}} \times 60$.
- `hourly_flow` = window-normalized `expected_flow`.
- $C$ = `capacity_vph`.
- $\alpha$ = configurable parameter (default: $0.15$).
- $\beta$ = configurable exponent (default: $4.0$).

### Properties
1. At zero flow ($V = 0$), $t = t_0$ (travel time equals free-flow travel time).
2. Below capacity ($V/C < 0.70$), travel time increases modestly.
3. Near and above capacity ($V/C \ge 1.0$), travel time increases nonlinearly, capturing congestion delays.
4. Fully configurable via `BPRParameters(alpha=..., beta=...)`.

---

## 6. Congestion Score Definition

The congestion score $S$ is mathematically defined as the **fraction of total travel time lost to congestion delay**:

$$S = \frac{t - t_0}{t} = 1.0 - \frac{t_0}{t}$$

Substituting the BPR equation:

$$S = \frac{\alpha (\text{hourly\_flow}/C)^\beta}{1 + \alpha (\text{hourly\_flow}/C)^\beta}$$

### Mathematical Properties
- **Bounded**: $0.0 \le S < 1.0$ for any finite non-negative flow.
- **Zero at Free Flow**: When $V = 0 \implies t = t_0 \implies S = 0.0$.
- **Monotonic**: Strictly increases as volume and delay increase.
- **Traffic-Model-Derived**: Derived directly from the physical delay curve; it is explicitly not an ungrounded or fabricated AI score.

---

## 7. Congestion Level Thresholds

Congestion states are classified using configurable Volume-to-Capacity thresholds (`CongestionThresholds`) aligned with standard transportation engineering Level of Service (LOS):

| Congestion Level | Default Utilization Condition ($V/C$) | Highway LOS Equivalent | Description |
|---|---|---|---|
| `FREE` | $V/C < 0.70$ | LOS A / B | Free-flow or light traffic, negligible delay. |
| `MODERATE` | $0.70 \le V/C < 0.90$ | LOS C | Stable traffic flow with perceptible slowing. |
| `HEAVY` | $0.90 \le V/C < 1.10$ | LOS D / E | Flow approaching or at design capacity, significant queuing. |
| `SEVERE` | $V/C \ge 1.10$ | LOS F | Oversaturated breakdown conditions, forced stop-and-go. |

Threshold boundaries are configurable via `CongestionThresholds(free_limit=..., moderate_limit=..., heavy_limit=...)` and validated to ensure strictly increasing positive thresholds.

---

## 8. Network Traffic Summary Semantics

The `NetworkTrafficSummary` produces network-wide aggregates. The semantics are explicitly defined as follows:

1. **`road_count`**: Total registered roads in the underlying `MobilityGraph` (network size).
2. **`active_roads_with_flow`**: Count of evaluated roads with strictly positive flow ($V > 0$).
3. **`evaluated_roads_count`**: Total road segments evaluated in the batch.
   > *Zero-Flow Policy*: `RoadFlow` records are never fabricated solely to represent zero-flow roads unless explicitly requested in Phase 2 aggregation.
4. **`average_utilization`**: Simple arithmetic mean across evaluated roads:
   $$\text{average\_utilization} = \frac{\sum_{i=1}^N \text{utilization\_ratio}_i}{N}$$
5. **`flow_weighted_utilization`**: Hourly-flow-weighted utilization:
   $$\text{flow\_weighted\_utilization} = \frac{\sum_{i=1}^N (\text{hourly\_flow}_i \times \text{utilization\_ratio}_i)}{\sum_{i=1}^N \text{hourly\_flow}_i}$$
   *Safe zero-flow rule*: When total hourly flow is $0$, defined safely as $0.0$ (no division by zero).
6. **`average_travel_time_minutes`**: Simple arithmetic mean of travel times across evaluated roads:
   $$\text{average\_travel\_time} = \frac{\sum_{i=1}^N t_i}{N}$$
7. **`flow_weighted_travel_time_minutes`**: Hourly-flow-weighted travel time:
   $$\text{flow\_weighted\_travel\_time} = \frac{\sum_{i=1}^N (\text{hourly\_flow}_i \times t_i)}{\sum_{i=1}^N \text{hourly\_flow}_i}$$
   *Safe zero-flow rule*: When total hourly flow is $0$, defined safely as $0.0$ (no division by zero).
8. **`congestion_level_counts`**: Dictionary of counts for each `CongestionLevel` among evaluated roads.

---

## 9. Time-Window Traffic Intelligence

Phase 3 preserves temporal metadata from Phase 2:
- Individual `TrafficMetric` records carry `time_window_start` and `time_window_end`.
- ISO date-time and time-only windows are validated; reversed, zero-duration, invalid, and partial windows are rejected.
- Flows without window metadata use the explicit calculator setting `default_aggregation_duration_hours` (default: 1 hour); no duration is inferred when a window is present.
- `TrafficMetricsCalculator.calculate_windowed_metrics()` takes the partitioned dictionary from Phase 2 (`ExpectedFlowAggregator.aggregate_by_time_window()`) and computes isolated metrics and network summaries for each discrete temporal window.
- This allows downstream systems to evaluate peak vs off-peak congestion transitions.

---

## 10. How Phase 4 Consumes Phase 3 Outputs

Phase 4 (Origin-Destination & Bottleneck Analysis) will consume:
1. **Bottleneck Identification**: Roads where `utilization_ratio >= 1.0` or `congestion_level == CongestionLevel.SEVERE` immediately identify primary capacity bottlenecks.
2. **Dynamic Travel Time Routing**: Phase 4 route choice and equilibrium models can replace static free-flow travel times with `estimated_travel_time_minutes` to re-estimate path travel times under congestion.
3. **Corridor Delay Aggregation**: Congestion scores and delays along multi-segment OD corridors can be aggregated to isolate systemic arterial bottlenecks.

---

## 11. Assumptions and Limitations

1. **Synthetic Validation**: Inputs are derived from the synthetic city network and mock probabilistic trajectories. No real-world accuracy or performance is claimed.
2. **Static Capacity Baseline**: Road capacity is assumed constant over the analysis window ($vph$). Temporal capacity drops due to weather or incidents are modeled via road closures or manual capacity overrides.
3. **Point-Queue / BPR Simplification**: The BPR formulation assumes link performance based on volume-to-capacity ratios. It does not model physical queue spillback across upstream junctions (which belongs to micro-simulation or Phase 6 counterfactuals).
4. **No Streaming/Database Dependency**: Batch computation is implemented using Python standard library and existing data structures. No database, Kafka, Redis, or cloud services are introduced.
