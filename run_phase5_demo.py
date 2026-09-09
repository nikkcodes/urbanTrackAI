"""Synthetic Phase 5 evidence-based network anomaly demonstration."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analytics.network import UrbanMobilityAnalyzer
from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import CandidateRoute, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.traffic.metrics import TrafficMetricsCalculator
from backend.anomaly.detector import AnomalyDetector
from backend.anomaly.models import MobilitySnapshot, RoadObservationStatus


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def build_phase4_state(graph: MobilityGraph, trajectories: list[NormalizedTrajectory], snapshot_id: str):
    flow_result = ExpectedFlowAggregator(graph).aggregate(
        trajectories,
        include_zero_flow_roads=False,
        time_window=("08:00:00", "08:15:00"),
    )
    metrics = TrafficMetricsCalculator(graph=graph).calculate_metrics(flow_result)
    analysis = UrbanMobilityAnalyzer().analyze(flow_result, metrics, graph, top_n=5)
    snapshot = MobilitySnapshot.from_phase4_result(
        snapshot_id=snapshot_id,
        result=analysis,
        graph=graph,
        aggregation_duration_hours=0.25,
        time_window_start="08:00:00",
        time_window_end="08:15:00",
        metadata={"source": "synthetic Phase 1-4 pipeline"},
    )
    return analysis, snapshot


def run_phase5_demo() -> None:
    """Compare normal and deliberately changed synthetic mobility states."""
    banner("UrbanTrackAI - Member 3 Network Anomaly Detection Demo (Phase 5)")
    print("Synthetic-data demonstration; this is evidence-based deterministic analytics, not ML or a probability.")

    graph = MobilityGraph.load_from_json(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
    trajectories = MockTrajectoryAdapter().adapt(PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json")
    baseline_trajectories = [trajectory for trajectory in trajectories if trajectory.time_window_start == "08:00:00"]

    changed_route = NormalizedTrajectory(
        track_id="CURRENT_ROUTE_SHIFT",
        origin_node="J01",
        destination_node="J12",
        candidate_routes=[CandidateRoute(["J01", "J04", "J08", "J10", "J12"], 1.0)],
        vehicle_weight=800.0,
        time_window_start="08:00:00",
        time_window_end="08:15:00",
    )
    current_trajectories = [
        replace(trajectory, vehicle_weight=trajectory.vehicle_weight * (40.0 if trajectory.track_id == "TRK_001" else 1.0))
        for trajectory in baseline_trajectories
        if trajectory.track_id != "TRK_005"
    ]
    current_trajectories.extend([
        changed_route,
        replace(next(trajectory for trajectory in baseline_trajectories if trajectory.track_id == "TRK_005"), vehicle_weight=600.0),
    ])

    _, baseline_snapshot = build_phase4_state(graph, baseline_trajectories, "baseline")
    _, current_snapshot = build_phase4_state(graph, current_trajectories, "current")

    # Simulate an observation omission only: this is not a zero-flow assertion.
    missing_current_flows = dict(current_snapshot.road_hourly_flows)
    omitted_road = "R01"
    missing_current_flows.pop(omitted_road, None)
    current_snapshot = replace(current_snapshot, road_hourly_flows=missing_current_flows)

    result = AnomalyDetector().compare(baseline_snapshot, current_snapshot, graph)
    print(f"\nBaseline/current window: {baseline_snapshot.time_window_start} - {baseline_snapshot.time_window_end}")
    print(f"Baseline roads observed: {len(baseline_snapshot.road_hourly_flows)}")
    print(f"Current roads observed: {len(current_snapshot.road_hourly_flows)}")

    banner("Road Flow Evidence")
    for evidence in result.road_evidence:
        if evidence.flow_signal or evidence.status != RoadObservationStatus.COMPARABLE:
            print(
                f"  {evidence.road_id}: status={evidence.status.value:<17} "
                f"baseline={evidence.baseline_hourly_flow!s:<8} current={evidence.current_hourly_flow!s:<8} "
                f"relative_change={evidence.relative_change!s:<8} signal={evidence.flow_signal or 'none'}"
            )
    print(f"  Explicit omission rule: {omitted_road} missing current != zero current flow.")

    banner("Distribution and Concentration Shifts")
    print(f"  OD Jensen-Shannon divergence: {result.od_divergence.divergence:.4f}")
    print(f"  Route Jensen-Shannon divergence: {result.route_divergence.divergence:.4f}")
    print(f"  HHI delta: {result.hhi_delta:+.4f}")
    print("  Bottleneck changes:")
    for change in result.bottleneck_changes[:10]:
        print(f"    {change.road_id}: {change.baseline_severity or 'MISSING'} -> {change.current_severity or 'MISSING'} ({change.change_type})")

    banner("Anomaly Events")
    for event in result.anomaly_events:
        print(f"  {event.anomaly_id} {event.anomaly_type}: severity={event.severity} evidence_count={event.evidence_count}")
        print(f"    signals={', '.join(event.triggered_signals)}")
        print(f"    explanation={event.explanation}")

    banner("Spatial Anomaly Regions")
    for region in result.spatial_regions:
        print(f"  {region.region_id}: roads={', '.join(region.road_ids)} severity={region.severity} evidence={region.evidence_count}")

    banner("Phase 5 Demo Complete - Evidence-Based Anomaly Detection Operational")


if __name__ == "__main__":
    run_phase5_demo()
