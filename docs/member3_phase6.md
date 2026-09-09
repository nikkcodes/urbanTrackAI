# Member 3 Phase 6: Counterfactual Network Simulation

## Purpose

Phase 6 answers a decision-support question:

> Given the observed mobility demand, what redistribution and congestion consequences does the model estimate when a road condition changes?

The implementation is a deterministic counterfactual network model. It does not claim to reproduce individual driver behavior, calibrated microscopic traffic, or real-world forecasting accuracy. It does not use ML, Bayesian inference, external routing APIs, or a traffic simulator.

## Inputs

The engine consumes:

- an existing `MobilityGraph`
- retained `NormalizedTrajectory` records from Phase 2
- a `Scenario` intervention
- an explicit aggregation duration and optional time-window metadata

Phase 4 and Phase 5 are optional upstream context. The core engine does not depend on their internals. A Phase 5 anomaly result can be used by a caller to choose roads to simulate, but is not required.

## Scenario Model

`Scenario` supports:

- `scenario_id`, `name`, and description
- closed road IDs
- positive capacity modifications in vehicles/hour
- positive speed modifications in km/hour

Unknown roads, duplicate closure IDs, empty IDs, non-finite values, and non-positive modifications are rejected. The current implementation focuses on closures and physical capacity/speed modifications; it does not simulate signal timing, incidents, or road construction.

## Counterfactual Graph Construction

`ScenarioGraphBuilder` calls `MobilityGraph.copy(deep=True)` and applies interventions only to the copy:

1. validate all referenced road IDs
2. close roads with the existing `close_road()` behavior
3. update copied capacity or speed attributes
4. update copied NetworkX edge weights for modified speeds

The baseline graph is never mutated. Multiple scenarios each receive independent graph copies.

## Route Reassignment Policy

Baseline assignments expand every candidate route in each trajectory using:

```text
assignment demand = vehicle_weight * candidate_route probability
```

This uses the existing Phase 2 probability semantics exactly once. It preserves uncertainty in the baseline instead of selecting a fake single route.

For the scenario:

1. preserve each baseline candidate route if its directed edges remain available
2. if that route is unavailable, find deterministic shortest simple alternatives using the existing graph route discovery
3. select the lowest free-flow travel-time route
4. break ties by route time, node sequence, and road-ID sequence
5. mark demand `UNROUTABLE` if no route remains

This is a deterministic scenario assignment, not a claim about exact individual driver choices.

## Demand Redistribution and Units

Assignment demand is accumulated expected vehicle weight for the aggregation window. The engine converts it to hourly flow exactly once:

```text
hourly assignment demand = assignment demand / aggregation_duration_hours
```

Road flow is accumulated back into `RoadFlow.expected_flow` using the same duration, then passed through the existing `TrafficMetricsCalculator`. Therefore:

- `expected_flow`: accumulated vehicles/vehicle-weight in the window
- `hourly_flow`: vehicles/hour
- `capacity_vph`: vehicles/hour
- utilization and BPR inputs remain unit-consistent

The engine includes every registered road in metric output, including closed roads with zero scenario demand, so impact comparisons can show a closure explicitly without changing Phase 2 behavior.

## Traffic Metrics Reuse

Scenario metrics use the existing Phase 3 `TrafficMetricsCalculator`, including its configured aggregation duration, utilization ratio, BPR travel time, congestion score, and congestion thresholds. Phase 6 does not implement a second congestion formula.

## Impact Comparison

`ScenarioComparator` produces:

- road hourly-flow, utilization, and travel-time deltas
- additional flow and new bottleneck flags
- baseline/scenario route and travel-time impacts for each demand assignment
- newly congested and relieved roads
- unroutable OD demand
- weighted network travel-time and utilization changes

Outputs are deterministically sorted and represented by serialization-friendly dataclasses.

## Unroutable Demand

When an intervention disconnects an origin-destination movement, the engine does not drop the demand. It returns `UnroutableDemand` with origin, destination, demand, and reason. Any positive unroutable demand makes the decision infeasible and unfavorable by default.

## Recommendation Classification

The decision summary uses explicit configurable thresholds:

- `UNFAVORABLE`: unroutable demand, increased travel time/utilization, or newly congested roads
- `FAVORABLE`: both weighted travel time and weighted utilization decrease beyond configured favorable thresholds
- `NEUTRAL`: neither condition is met

This is a transparent decision rule, not an opaque score.

## Example

Closing a road removes its edge from a copied graph. Candidate demand using that road is reassigned to an available route where possible. The result shows the closed road's lost flow, the alternative roads' additional hourly flow, changed utilization and BPR travel times, route changes, and any new heavy/severe roads.

`run_phase6_demo.py` runs closure and speed scenarios against the existing synthetic city network and mock trajectories only.

## Phase 5 Relationship

Phase 5 can identify anomalous or severe roads that deserve counterfactual evaluation. Phase 6 can accept the selected road as a `Scenario`, but does not require or modify `AnomalyAnalysisResult`.

## Limitations and Future Extensions

This is a deterministic graph-based decision engine intended for hackathon decision support. It does not model queues, spillback, capacity equilibrium, signal timing, stochastic route choice, driver adaptation, or calibrated real-world conditions. Future work may add richer assignment policies or scenario types while preserving the current baseline/scenario result contracts.
