# Member 3 Phase 5: Evidence-Based Network Anomaly Detection

## Purpose

Phase 5 compares two compatible mobility states and detects explainable changes across independent signals. It is deterministic statistical analytics, not machine learning, Bayesian inference, or a probability score.

Signals include:

- comparable-road hourly-flow changes
- OD demand distribution change
- probabilistic route/corridor redistribution
- Phase 4 hourly-flow HHI change
- Phase 4 bottleneck severity/state changes
- topology-based grouping of anomalous roads

The engine consumes Phase 4 outputs through `MobilitySnapshot.from_phase4_result()` and does not recalculate Phase 4 analytics.

## Inputs and Snapshot Contract

`MobilitySnapshot` contains only Phase 5 inputs:

- snapshot identity and road-network identity
- aggregation duration in hours
- road `hourly_flow` values in vehicles/hour
- accumulated OD demand by `(origin, destination)`
- accumulated probabilistic route demand by road-ID tuple
- Phase 4 HHI
- Phase 4 bottleneck records
- optional ISO time-window metadata

Phase 2 `expected_flow` is never compared directly with Phase 3 `hourly_flow`. OD and route demand remain accumulated values for their comparable window. HHI is unitless.

Baseline and current snapshots must have the same topology identity, compatible aggregation duration, and identical time-window metadata. Incompatible snapshots raise a validation error rather than being silently converted or merged.

## Missing Is Not Zero

Road observations are compared only when both snapshots contain the road key. If a road is absent:

- baseline present, current absent: `MISSING_CURRENT`
- baseline absent, current present: `MISSING_BASELINE`

Missing observations have no absolute or relative flow change and do not produce a flow surge/drop. The engine never fabricates zero-flow observations to equalize snapshots.

## Road Flow Evidence

For comparable roads:

```text
absolute_change = current_hourly_flow - baseline_hourly_flow
relative_change = absolute_change / max(abs(baseline_hourly_flow), epsilon)
```

A configurable absolute-change threshold protects near-zero baselines. Configurable relative and absolute thresholds produce `FLOW_SURGE` or `FLOW_DROP`. Each `RoadAnomalyEvidence` record exposes the baseline/current rate, changes, status, signal, and evidence count.

## OD and Route Distribution Shifts

The engine uses Phase 4 OD demand directly; it does not reconstruct OD from road flow. Phase 4 route demand already represents `vehicle_weight * route_probability`, so Phase 5 does not apply route probabilities again.

Both distributions are safely normalized over their comparable window. Missing keys receive zero mass. Jensen-Shannon divergence is calculated with base-2 logarithms, so it is symmetric and bounded. Empty versus empty is `0.0`; empty versus non-empty is handled deterministically without NaN. Configurable thresholds trigger `OD_SHIFT` and `ROUTE_REDISTRIBUTION`.

## HHI Shift

Phase 4 HHI is reused directly. The detector exposes:

```text
hhi_delta = current_hhi - baseline_hhi
```

The configurable absolute delta threshold triggers `NETWORK_CONCENTRATION_SHIFT`; no alternate HHI definition is introduced.

## Bottleneck Shift

Phase 4 bottleneck severity labels are compared by the existing order `FREE < MODERATE < HEAVY < SEVERE`. The detector reports deterministic changes such as:

- `NEW_BOTTLENECK`
- `SEVERITY_INCREASE`
- `SEVERITY_DECREASE`
- `BOTTLENECK_DISAPPEARED`

No predictive bottleneck model or capacity/closure simulation is used.

## Multi-Signal Consensus

Each signal is retained independently. The detector emits individual signal events and, when at least one signal is present, a `MOBILITY_STATE_CHANGE` consensus event. Default severity rules are evidence-based classifications:

- `NORMAL`: no signal
- `LOW`: one moderate signal
- `MEDIUM`: one strong signal or at least two moderate signals
- `HIGH`: at least two strong signals or at least three signals
- `CRITICAL`: at least three strong signals

These are configurable deterministic rules. Severity is not a statistical probability.

## Spatial Grouping

`SpatialAnomalyGrouper` uses the existing directed `MobilityGraph`. Anomalous roads sharing a `from_node` or `to_node` are grouped into connected components. Results are sorted deterministically and returned as `SpatialAnomalyRegion` records. No GIS library or geographic clustering is added.

## Outputs

`AnomalyAnalysisResult` contains:

- baseline/current snapshot IDs
- deterministic road evidence
- OD and route divergence records with normalized distributions
- HHI delta
- bottleneck state changes
- individual and consensus anomaly events
- topology-based spatial anomaly regions

The models are serialization-friendly for future consumers while remaining independent of an API or UI.

## Synthetic Demo

`run_phase5_demo.py` builds a baseline from the existing synthetic Phase 1–4 pipeline, constructs a deliberately changed current state with increased corridor demand and route redistribution, removes one current road observation, and prints flow evidence, both divergences, HHI change, bottleneck changes, consensus evidence, and spatial regions. No real-world accuracy is claimed.

## Limitations and Phase 6

The detector compares supplied snapshots; it does not forecast traffic, infer causality, or model sensor reliability beyond distinguishing missing observations. HHI and distribution comparisons are descriptive. Phase 6 may consume `AnomalyAnalysisResult` for counterfactual closure or accident redistribution simulation. Phase 6 is intentionally not implemented here.
