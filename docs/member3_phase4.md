# Member 3 Phase 4: Urban Mobility Intelligence

## Purpose

Phase 4 consumes the existing Phase 2 trajectory/flow contract and Phase 3 traffic metrics to produce descriptive city mobility analytics:

- accumulated origin-destination demand
- probabilistic route and corridor demand
- deterministic bottleneck rankings
- hourly road-flow concentration and shares
- graph-based structural importance
- transparent road-priority rankings

This phase is analytics only. It does not simulate road closures, redistribute traffic, forecast future traffic, expose an API, or provide a dashboard.

## Architecture

```text
Member 2 trajectories
        |
        v
Phase 2 NormalizedTrajectory / FlowAggregationResult
        |
        v
Phase 3 TrafficMetric records
        |
        v
+----------------+----------------+-------------------+
| OD Analysis    | Bottlenecks    | Network Analytics |
+----------------+----------------+-------------------+
        |
        v
Urban Mobility Insights
        |
        v
Phase 5 anomaly detection and Phase 6 counterfactual simulation
```

`ODAnalyzer` reads `FlowAggregationResult.trajectories` directly. Road flow alone cannot reconstruct origin and destination demand.

## OD Methodology

For each retained `NormalizedTrajectory`:

```text
OD demand(origin, destination, window) += vehicle_weight
```

Candidate-route probabilities are not applied to OD demand a second time. They already describe route uncertainty within a trajectory whose origin, destination, and vehicle weight are known.

`ODPairDemand.demand` is an accumulated expected vehicle/vehicle-weight count for its preserved aggregation window. It is not vehicles/hour. `ODMatrix` keeps records separated by `(time_window_start, time_window_end)`. A lookup across multiple windows must specify a window; global ranking returns all preserved records in deterministic order.

## Probabilistic Route Demand

For every candidate route:

```text
route demand += vehicle_weight * route_probability
```

Routes remain separate and are ranked without collapsing the candidate distribution. Existing `ExpectedFlowAggregator` validation and node-to-road mapping are reused, keeping the Phase 4 adapter boundary compatible with future Member 2 implementations.

## Bottlenecks

`BottleneckDetector` ranks every supplied Phase 3 `TrafficMetric` using an explicit additive score:

```text
bottleneck_score =
    utilization_weight * utilization_ratio
  + congestion_weight * congestion_score
  + delay_weight * (delay_minutes / free_flow_time_minutes)
```

Weights are configurable, finite, and non-negative; at least one must be positive. Default weights are all `1.0`. The score is a deterministic ranking of measured signals, not an AI prediction. Severity retains the Phase 3 congestion level, and each record exposes hourly flow, capacity, delay, score, and signal labels.

## Network Intelligence

### Flow concentration

Concentration uses Phase 3 `hourly_flow`, never accumulated `expected_flow`:

```text
flow_share_i = hourly_flow_i / sum(hourly_flow)
HHI = sum(flow_share_i ** 2)
```

Zero total hourly flow returns HHI `0.0` and no positive-flow shares. With positive flow, shares sum to `1.0` over evaluated active roads.

### Structural importance

The first structural metric is normalized NetworkX directed edge betweenness centrality on the active `MobilityGraph`. It describes how often an edge lies on shortest paths in the current observed topology. Phase 4 does not remove roads or simulate redistribution.

### Priority ranking

`NetworkIntelligenceAnalyzer.rank_priority_roads()` returns the underlying flow share, utilization, congestion, delay ratio, and structural importance beside a deterministic configurable additive priority score:

```text
priority_score =
    w_flow_share * flow_share
  + w_utilization * utilization_ratio
  + w_congestion * congestion_score
  + w_delay * delay_ratio
  + w_structural * structural_importance
```

This is a descriptive prioritization aid, not a Bayesian model or predictive forecast.

## Units and Time Windows

- Phase 2 `expected_flow`: accumulated expected vehicle/vehicle-weight traversals over a window.
- Phase 3 `hourly_flow`: `expected_flow / aggregation_duration_hours`, in vehicles/hour.
- Phase 1 `capacity_vph`: vehicles/hour.
- Phase 4 OD demand: accumulated trajectory weight over a preserved window.
- Phase 4 route demand: accumulated expected route weight over a preserved window.
- Phase 4 HHI and road shares: based on `hourly_flow`.

Phase 4 reuses the existing ISO time-only and ISO datetime conventions. Invalid, partial, reversed, zero-duration, or incompatible windows are rejected in Phase 4 demand models. Multiple windows are preserved rather than silently merged.

## Limitations and Later Phases

Synthetic fixtures validate deterministic behavior only; they do not establish real-world accuracy. Edge betweenness uses the current active graph and does not model demand-dependent route choice. Bottleneck and priority scores are analytical rankings, not forecasts.

Phase 5 may consume these records for anomaly detection. Phase 6 may consume them for counterfactual road-closure and accident redistribution simulation. Those capabilities are intentionally outside Phase 4.
