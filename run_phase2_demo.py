"""Demonstration and smoke test for UrbanTrackAI Phase 2 Flow Aggregation Engine.

Demonstrates:
1. Loading the Phase 1 synthetic city network (MobilityGraph).
2. Loading mock probabilistic vehicle trajectories.
3. Adapting mock data into internal NormalizedTrajectory representations.
4. Aggregating expected road flows across directed road segments.
5. Displaying deterministic, uncertainty-preserving flow results.
6. Demonstrating time-window flow aggregation and trajectory retention for Phase 4.
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


def print_banner(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def run_phase2_demo() -> None:
    """Execute the Phase 2 expected road flow aggregation demonstration."""
    print_banner("UrbanTrackAI - Member 3 Flow Aggregation Engine Demo (Phase 2)")

    # 1. Load synthetic network
    city_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
    print(f"[1] Loading synthetic city network from:\n    {city_path}")

    if not city_path.exists():
        print(f"ERROR: Dataset not found at {city_path}")
        sys.exit(1)

    graph = MobilityGraph.load_from_json(city_path)
    print(f"    Network loaded: {graph.name}")
    print(f"    Nodes: {graph.node_count} | Active Roads: {graph.active_road_count}")

    # 2. Load mock probabilistic trajectories via Adapter boundary
    mock_path = PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json"
    print(f"\n[2] Loading mock probabilistic trajectories via MockTrajectoryAdapter from:\n    {mock_path}")

    if not mock_path.exists():
        print(f"ERROR: Mock trajectories fixture not found at {mock_path}")
        sys.exit(1)

    adapter = MockTrajectoryAdapter()
    trajectories = adapter.adapt(mock_path)
    print(f"    Successfully adapted {len(trajectories)} normalized trajectories:")

    for idx, t in enumerate(trajectories, 1):
        window_str = f" [{t.time_window_start} - {t.time_window_end}]" if t.time_window_start else ""
        print(f"    - Trajectory #{idx} [{t.track_id}]: {t.origin_node} -> {t.destination_node} | "
              f"Weight: {t.vehicle_weight:.1f} | Routes: {len(t.candidate_routes)}{window_str}")
        for r_idx, r in enumerate(t.candidate_routes, 1):
            path_str = " -> ".join(r.nodes)
            print(f"        Route {r_idx} (p={r.probability:.2f}): {path_str}")

    # 3. Expected Road Flow Aggregation (Overall)
    print_banner("Aggregating Expected Road Flows (Uncertainty-Preserved)")
    aggregator = ExpectedFlowAggregator(graph=graph)
    result = aggregator.aggregate(trajectories, include_zero_flow_roads=False)

    print("Road-level aggregated flows (active traffic corridors):")
    print(f"{'Road ID':<10} {'Segment':<18} {'Expected Flow':<16} {'Contributing Trajectories':<26}")
    print("-" * 75)

    all_flows = result.all_flows(sorted_by_id=True)
    total_vehicle_miles_proxy = 0.0

    for rf in all_flows:
        segment_str = f"{rf.from_node} -> {rf.to_node}"
        road_obj = graph.get_road(rf.road_id)
        dist_km = road_obj.distance_km if road_obj else 0.0
        total_vehicle_miles_proxy += rf.expected_flow * dist_km

        print(f"{rf.road_id:<10} {segment_str:<18} {rf.expected_flow:>8.2f} veh      "
              f"{rf.contributing_trajectories_count:>3} track(s)")

    print("-" * 75)
    print(f"Summary: {len(all_flows)} roads receiving expected flow from {len(trajectories)} trajectories.")

    # 4. Partitioned Time-Window Aggregation
    print_banner("Time-Window Analysis: Partitioned Road Flows")
    windowed_results = aggregator.aggregate_by_time_window(trajectories, include_zero_flow_roads=False)

    for (start, end), w_res in windowed_results.items():
        label = f"{start} - {end}" if start and end else "Unspecified Window"
        w_flows = w_res.all_flows(sorted_by_id=True)
        print(f"\n>> Time Window: {label} ({len(w_res.trajectories)} trajectories, {len(w_flows)} active roads)")
        for rf in w_flows:
            print(f"   Road {rf.road_id} ({rf.from_node} -> {rf.to_node}): expected flow = {rf.expected_flow:.2f} veh")

    # 5. Phase 3 & Phase 4 Downstream Readiness Check
    print_banner("Downstream Readiness Verification")
    print("1. Phase 3 Traffic Volume & Congestion Interface:")
    print("   - Road flows are isolated, deterministic, and mapped directly to road IDs.")
    print("   - Ready to compute V/C utilization and BPR travel times in Phase 3.")
    print(f"\n2. Phase 4 Origin-Destination Analysis Interface:")
    print(f"   - Aggregation result retained {len(result.trajectories)} trajectory objects with OD information intact.")
    print("   - Sample OD Pair from retained data: "
          f"Track {result.trajectories[0].track_id} ({result.trajectories[0].origin_node} -> "
          f"{result.trajectories[0].destination_node})")

    print_banner("Phase 2 Demo Complete - Flow Aggregation Operational")


if __name__ == "__main__":
    run_phase2_demo()
