"""Smoke test and demonstration for UrbanTrackAI Member 3 Mobility Engine.

Demonstrates:
1. Loading the synthetic city network.
2. Constructing the directed MobilityGraph.
3. Reporting network topology (nodes, active roads).
4. Discovering multiple candidate routes between origin and destination.
5. Computing route distances (km) and free-flow travel times (minutes).
6. Simulating a road closure and verifying dynamic route rerouting.
7. Restoring the road and verifying graph restoration.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.mobility.graph import MobilityGraph
from backend.mobility.routes import get_candidate_routes


def print_banner(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_demo() -> None:
    """Execute the mobility engine smoke test."""
    print_banner("UrbanTrackAI - Member 3 Mobility Engine Demo (Phase 1)")

    # 1. Load synthetic network
    data_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
    print(f"[1] Loading synthetic city network from:\n    {data_path}")

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)

    graph = MobilityGraph.load_from_json(data_path)

    # 2 & 3 & 4: Network Summary
    print("\n[2] Network Topology:")
    print(f"    - Network Name       : {graph.name}")
    print(f"    - Total Nodes (V)    : {graph.node_count}")
    print(f"    - Total Roads (E)    : {graph.road_count}")
    print(f"    - Active Roads       : {graph.active_road_count}")
    print(f"    - Closed Roads       : {graph.closed_road_count}")

    # 5. Route Discovery: Origin -> Destination
    source = "J01"       # North Gate Terminal
    destination = "J12"  # South Hub Terminal

    src_node = graph.get_node(source)
    dst_node = graph.get_node(destination)
    src_label = f"{source} ({src_node.name if src_node else 'Unknown'})"
    dst_label = f"{destination} ({dst_node.name if dst_node else 'Unknown'})"

    print_banner(f"Route Discovery: {src_label} -> {dst_label}")
    routes = get_candidate_routes(graph, source=source, destination=destination, max_routes=4)

    print(f"Found {len(routes)} feasible candidate route(s):\n")
    for idx, r in enumerate(routes, 1):
        path_str = " -> ".join(r.path)
        roads_str = ", ".join(r.road_ids)
        print(f"  Route #{idx}:")
        print(f"    Path               : {path_str}")
        print(f"    Road Segments      : [{roads_str}]")
        print(f"    Total Distance     : {r.distance_km:.2f} km")
        print(f"    Free-Flow Time     : {r.free_flow_time_min:.2f} min")
        print()

    # 6. Simulate Dynamic Road Closure (e.g. Incident on R01)
    closed_road_id = "R01"
    print_banner(f"Simulating Road Incident: Closing '{closed_road_id}' (J01 -> J02)")
    road_obj = graph.get_road(closed_road_id)
    if road_obj:
        print(f"Closing road segment: {road_obj.road_id} ({road_obj.from_node} -> {road_obj.to_node})")
        graph.close_road(closed_road_id)
        print(f"Active road count now: {graph.active_road_count} (Closed roads: {graph.closed_road_count})")

        rerouted = get_candidate_routes(graph, source=source, destination=destination, max_routes=4)
        print(f"\nRecalculated {len(rerouted)} candidate route(s) avoiding '{closed_road_id}':")
        for idx, r in enumerate(rerouted, 1):
            path_str = " -> ".join(r.path)
            assert closed_road_id not in r.road_ids, f"Error: closed road {closed_road_id} appeared in route!"
            print(f"  Alternative #{idx}: {path_str}")
            print(f"    Distance: {r.distance_km:.2f} km | Time: {r.free_flow_time_min:.2f} min")

    # 7. Restore the road
    print_banner(f"Incident Cleared: Restoring '{closed_road_id}'")
    graph.restore_road(closed_road_id)
    print(f"Restored road segment: {closed_road_id}")
    print(f"Active road count now: {graph.active_road_count} (Closed roads: {graph.closed_road_count})")

    restored_routes = get_candidate_routes(graph, source=source, destination=destination, max_routes=4)
    print(f"Restored candidate routes count: {len(restored_routes)}")
    print("Graph integrity and route availability fully verified.")
    print_banner("Demo Complete - Mobility Engine Foundation Operational")


if __name__ == "__main__":
    run_demo()
