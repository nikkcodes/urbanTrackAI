"""Synthetic Phase 4 urban mobility intelligence demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analytics.network import UrbanMobilityAnalyzer
from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.mobility.graph import MobilityGraph
from backend.traffic.metrics import TrafficMetricsCalculator


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def run_phase4_demo() -> None:
    """Run Phase 1 through Phase 4 analytics using synthetic data only."""
    banner("UrbanTrackAI - Member 3 Urban Mobility Intelligence Demo (Phase 4)")
    print("Synthetic-data demonstration; results are algorithm validation, not real-world accuracy claims.")

    graph = MobilityGraph.load_from_json(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
    trajectories = MockTrajectoryAdapter().adapt(PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json")
    flow_result = ExpectedFlowAggregator(graph).aggregate(trajectories)
    metrics = TrafficMetricsCalculator(graph=graph).calculate_metrics(flow_result)
    result = UrbanMobilityAnalyzer().analyze(flow_result, metrics, graph, top_n=5)

    print(f"\nTrajectories retained: {result.od_analysis.trajectory_count}")
    print(f"Trajectory vehicle weight: {sum(t.vehicle_weight for t in trajectories):.2f}")
    print(f"Registered roads: {result.network.road_count} | Evaluated roads: {result.network.evaluated_roads_count}")

    banner("Top OD Demand (accumulated trajectory weight by window)")
    for pair in result.od_analysis.top_od_pairs[:5]:
        window = f"{pair.time_window_start or 'unspecified'} - {pair.time_window_end or 'unspecified'}"
        print(f"  {pair.origin} -> {pair.destination}: {pair.demand:.2f} expected vehicles | {window}")

    banner("Top Probabilistic Route Demand")
    for route in sorted(result.route_demands, key=lambda item: (-item.demand, item.route_nodes))[:5]:
        print(f"  {' -> '.join(route.route_nodes)}: {route.demand:.2f} expected route demand")

    banner("Top Bottleneck Candidates")
    for bottleneck in result.bottlenecks[:5]:
        print(
            f"  {bottleneck.road_id}: severity={bottleneck.severity:<8} "
            f"score={bottleneck.bottleneck_score:.4f} u={bottleneck.utilization_ratio:.4f} "
            f"hourly_flow={bottleneck.hourly_flow:.2f} delay={bottleneck.delay_minutes:.4f} min"
        )

    banner("Hourly Road Flow Shares")
    for road_id, share in sorted(result.network.road_flow_shares.items(), key=lambda item: (-item[1], item[0]))[:10]:
        print(f"  {road_id}: {share:.4f}")
    print(f"HHI concentration: {result.network.flow_concentration_hhi:.4f}")

    banner("Structurally Important Active Roads")
    structural = sorted(result.network.structural_importance.items(), key=lambda item: (-item[1], item[0]))[:5]
    for road_id, score in structural:
        print(f"  {road_id}: normalized edge betweenness={score:.4f}")

    banner("Top Mobility-Priority Roads")
    for priority in result.top_priority_roads:
        print(
            f"  {priority.road_id}: priority={priority.priority_score:.4f} "
            f"share={priority.flow_share:.4f} u={priority.utilization_ratio:.4f} "
            f"congestion={priority.congestion_score:.4f} structural={priority.structural_importance:.4f}"
        )

    banner("Phase 4 Demo Complete - Descriptive Mobility Intelligence Operational")


if __name__ == "__main__":
    run_phase4_demo()
