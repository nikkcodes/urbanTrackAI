"""Demonstration and smoke test for UrbanTrackAI Phase 3 Traffic Intelligence Engine.

Demonstrates:
1. Loading the Phase 1 synthetic city network (MobilityGraph).
2. Loading Phase 2 mock probabilistic trajectories via MockTrajectoryAdapter.
3. Running Phase 2 ExpectedFlowAggregator to compute expected road flows.
4. Feeding RoadFlow records into Phase 3 TrafficMetricsCalculator.
5. Computing capacity utilization ratios, BPR congested travel times,
   normalized congestion scores, and discrete congestion levels.
6. Displaying a readable segment-level traffic metrics report.
7. Generating a network-level summary with flow-weighted statistics.
8. Demonstrating time-window traffic metrics across temporal slices.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.mobility.graph import MobilityGraph
from backend.traffic.metrics import TrafficMetricsCalculator
from backend.traffic.models import BPRParameters, CongestionThresholds


def print_banner(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_phase3_demo() -> None:
    """Execute the Phase 3 traffic intelligence demonstration."""
    print_banner("UrbanTrackAI - Member 3 Traffic Intelligence Engine Demo (Phase 3)")

    # 1. Load Phase 1 synthetic city network
    city_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
    print(f"[1] Loading Phase 1 synthetic city network from:\n    {city_path}")

    if not city_path.exists():
        print(f"ERROR: Dataset not found at {city_path}")
        sys.exit(1)

    graph = MobilityGraph.load_from_json(city_path)
    print(f"    Network: {graph.name}")
    print(f"    Total Nodes: {graph.node_count} | Total Registered Roads: {graph.road_count}")

    # 2. Load Phase 2 mock trajectories
    mock_path = PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json"
    print(f"\n[2] Loading Phase 2 mock trajectories from:\n    {mock_path}")

    if not mock_path.exists():
        print(f"ERROR: Mock trajectories fixture not found at {mock_path}")
        sys.exit(1)

    adapter = MockTrajectoryAdapter()
    trajectories = adapter.adapt(mock_path)
    print(f"    Adapted {len(trajectories)} normalized probabilistic trajectories.")
    print(f"    Total trajectory vehicle weight: {sum(t.vehicle_weight for t in trajectories):.2f}")

    # 3. Execute Phase 2 Expected Flow Aggregation
    print("\n[3] Executing Phase 2 Expected Flow Aggregation...")
    flow_aggregator = ExpectedFlowAggregator(graph=graph)
    flow_result = flow_aggregator.aggregate(trajectories, include_zero_flow_roads=False)
    print(f"    Aggregated flows across {len(flow_result.flows)} active road segments.")

    # 4. Initialize Phase 3 Traffic Intelligence Engine
    print("\n[4] Initializing Phase 3 TrafficMetricsCalculator...")
    bpr_params = BPRParameters(alpha=0.15, beta=4.0)
    thresholds = CongestionThresholds(free_limit=0.70, moderate_limit=0.90, heavy_limit=1.10)
    calculator = TrafficMetricsCalculator(
        graph=graph,
        bpr_params=bpr_params,
        thresholds=thresholds,
    )
    print(f"    BPR Parameters: alpha={bpr_params.alpha}, beta={bpr_params.beta}")
    print(f"    Congestion Thresholds: FREE < {thresholds.free_limit} <= MODERATE < "
          f"{thresholds.moderate_limit} <= HEAVY < {thresholds.heavy_limit} <= SEVERE")

    # 5. Compute segment-level traffic metrics
    print_banner("Road-Level Traffic Intelligence Report")
    metrics = calculator.calculate_metrics(flow_result)

    headers = (
        f"{'Road ID':<8} {'Segment Trips':<14} {'Flow (vph)':<12} {'Cap (vph)':<10} {'V/C Ratio':<10} "
        f"{'t0 (min)':<9} {'t_est (min)':<12} {'Score':<8} {'Level':<10}"
    )
    print(headers)
    print("-" * 80)

    for m in metrics:
        row = (
            f"{m.road_id:<8} {m.expected_flow:>12.2f}  {m.hourly_flow:>10.2f}  {m.capacity_vph:>8.0f}  "
            f"{m.utilization_ratio:>8.4f}  {m.free_flow_time_minutes:>7.2f}  "
            f"{m.estimated_travel_time_minutes:>10.2f}  {m.congestion_score:>6.4f}  "
            f"{m.congestion_level.value:<10}"
        )
        print(row)

    print("-" * 80)

    # 6. Network-Level Summary
    print_banner("Network-Level Traffic Performance Summary")
    summary = calculator.summarize_network(metrics)

    print(f"  Total Network Roads (MobilityGraph) : {summary.road_count}")
    print(f"  Evaluated Active Corridors          : {summary.evaluated_roads_count}")
    print(f"  Active Roads with Flow (> 0 veh)    : {summary.active_roads_with_flow}")
    print(f"  Expected Vehicle-Segment Traversals : {summary.total_expected_flow:.2f} traversals")
    print()
    print(f"  Average Utilization (Simple Mean)   : {summary.average_utilization:.4f}")
    print(f"  Hourly-Flow-Weighted Utilization    : {summary.flow_weighted_utilization:.4f}")
    print()
    print(f"  Average Travel Time (Simple Mean)   : {summary.average_travel_time_minutes:.2f} min")
    print(f"  Hourly-Flow-Weighted Travel Time    : {summary.flow_weighted_travel_time_minutes:.2f} min")
    print()
    print("  Congestion State Distribution:")
    for level, count in sorted(summary.congestion_level_counts.items()):
        bar = "#" * count
        print(f"    - {level:<10}: {count:>2} road(s) {bar}")

    # 7. Time-Window Traffic Intelligence
    print_banner("Time-Window Partitioned Traffic Intelligence")
    windowed_flows = flow_aggregator.aggregate_by_time_window(trajectories, include_zero_flow_roads=False)
    windowed_metrics = calculator.calculate_windowed_metrics(windowed_flows)

    for (start, end), (w_metrics, w_summary) in windowed_metrics.items():
        w_label = f"{start} - {end}" if start and end else "Unspecified Window"
        print(f"\n>> Analysis Window: {w_label}")
        print(f"   Active Roads: {w_summary.evaluated_roads_count} | Expected Segment Traversals: {w_summary.total_expected_flow:.2f}")
        print(
            f"   Hourly-Flow-Weighted V/C: {w_summary.flow_weighted_utilization:.4f} | "
            f"Hourly-Flow-Weighted Travel Time: {w_summary.flow_weighted_travel_time_minutes:.2f} min"
        )
        print(f"   Congestion Levels: {w_summary.congestion_level_counts}")

    print_banner("Phase 3 Demo Complete - Traffic Intelligence Operational")


if __name__ == "__main__":
    run_phase3_demo()
