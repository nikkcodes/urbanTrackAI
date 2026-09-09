"""Synthetic Phase 6 counterfactual network simulation demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.flow.adapters import MockTrajectoryAdapter
from backend.mobility.graph import MobilityGraph
from backend.simulation.engine import CounterfactualSimulationEngine
from backend.simulation.models import Scenario


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def run_phase6_demo() -> None:
    """Run deterministic closure and speed scenarios on the existing synthetic city."""
    banner("UrbanTrackAI - Member 3 Counterfactual Simulation Demo (Phase 6)")
    print("Synthetic-data decision-support model; results are estimated redistribution, not exact prediction.")

    graph = MobilityGraph.load_from_json(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
    trajectories = MockTrajectoryAdapter().adapt(PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json")
    trajectories = [trajectory for trajectory in trajectories if trajectory.time_window_start == "08:00:00"]
    engine = CounterfactualSimulationEngine(
        aggregation_duration_hours=0.25,
        time_window_start="08:00:00",
        time_window_end="08:15:00",
    )
    scenarios = (
        Scenario("close_R01", "Close R01", "Counterfactual closure of R01", closed_road_ids=("R01",)),
        Scenario("slow_R01", "Slow R01", "Counterfactual speed reduction on R01", speed_modifications_kmph={"R01": 20.0}),
    )
    results = engine.simulate_many(graph, trajectories, scenarios)

    print(f"\nBaseline retained trajectories: {len(trajectories)}")
    print(f"Baseline graph roads: {graph.road_count} | Baseline R01 closed: {graph.is_road_closed('R01')}")
    for result in results:
        banner(f"Scenario: {result.scenario.name}")
        print(f"Feasible: {result.decision.feasible}")
        print(f"Recommendation: {result.decision.recommendation_class.value}")
        print(f"Rerouted demand: {result.decision.rerouted_demand:.2f} vehicles in window")
        print(f"Unroutable demand: {result.decision.unroutable_demand:.2f} vehicles in window")
        print(f"Weighted travel-time delta: {result.decision.total_travel_time_delta_minutes:+.4f} min")
        print(f"Weighted utilization delta: {result.decision.network_congestion_delta:+.6f}")
        print(f"Newly congested roads: {', '.join(result.decision.newly_congested_roads) or 'none'}")
        print(f"Relieved roads: {', '.join(result.decision.relieved_roads) or 'none'}")
        print(f"Decision explanation: {result.decision.explanation}")
        print("Road impacts:")
        for impact in result.road_impacts:
            if abs(impact.flow_delta) > 1e-9 or impact.became_new_bottleneck:
                print(
                    f"  {impact.road_id}: {impact.baseline_hourly_flow:.2f} -> "
                    f"{impact.scenario_hourly_flow:.2f} veh/h "
                    f"delta={impact.flow_delta:+.2f} u={impact.scenario_utilization:.4f}"
                )
        print("Route changes:")
        for impact in result.od_impacts:
            if impact.route_changed or impact.status == "UNROUTABLE":
                print(f"  {impact.origin} -> {impact.destination}: {impact.status} {impact.baseline_route} -> {impact.scenario_route}")

    print(f"\nBaseline graph unchanged after all scenarios: {not graph.is_road_closed('R01')}")
    banner("Phase 6 Demo Complete - Counterfactual Decision Support Operational")


if __name__ == "__main__":
    run_phase6_demo()
