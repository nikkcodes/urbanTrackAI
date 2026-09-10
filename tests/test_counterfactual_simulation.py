"""
Unit and Integration Tests for Day 8 Counterfactual Traffic Simulation.

Comprehensive verification of:
1. Scenario validation and error rejection (INVALID_INPUT).
2. Strict baseline immutability (RoadGraph, trajectories, candidate routes, probabilities).
3. Probability semantics (original_route_probability vs counterfactual_allocation_probability).
4. Demand semantics (vehicle_weight * allocation, NEVER multiplied by reliability).
5. Demand conservation (sum(counterfactual_allocation_probability) == 1.0).
6. Road closure handling (closed roads receive zero counterfactual flow).
7. Displaced demand and alternate corridor identification.
8. Capacity changes (capacity reduction and increase impacts on utilization without fake rerouting).
9. Demand changes (demand_multiplier scaling).
10. Multiple closures applied jointly.
11. No-alternative route handling (unroutable_demand explicitly reported, zero fake routes).
12. Ambiguous route renormalization.
13. Evidence reliability preservation (reliability != physical demand).
14. Road recovery (restoration of previously closed roads).
15. Temporal safety (missing time windows remain uncalibrated without fake hourly rates).
16. Edge cases (empty trajectories, zero demand, disconnected network).
"""

import copy
import json
import math
from pathlib import Path
import unittest

from inference.road_graph import RoadGraph, RoadNode, RoadEdge
from schemas.normalized_trajectory_schema import NormalizedCandidateRoute, NormalizedTrajectory
from schemas.mobility_schema import RoadStatus
from schemas.scenario_schema import (
    CounterfactualReport,
    ScenarioDefinition,
    ScenarioStatus,
    ScenarioType,
)
from simulation.counterfactual_engine import CounterfactualEngine
from simulation.impact_analyzer import ImpactAnalyzer


class TestCounterfactualSimulation(unittest.TestCase):
    """Test suite for UrbanTrack AI Day 8 Counterfactual Simulation Engine."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load synthetic network and scenarios for testing."""
        base_dir = Path(__file__).resolve().parent.parent
        network_path = base_dir / "data" / "synthetic" / "city_network.json"
        scenarios_path = base_dir / "data" / "synthetic" / "day8_counterfactual_scenarios.json"

        cls.graph = RoadGraph.from_json_file(network_path)

        with open(scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cls.scenarios_raw = data["scenarios"]

        cls.engine = CounterfactualEngine()

    def setUp(self) -> None:
        """Reset a fresh copy of the road graph before each test."""
        self.test_graph = copy.deepcopy(self.graph)

    # -------------------------------------------------------------------------
    # 1. SCENARIO VALIDATION TESTS
    # -------------------------------------------------------------------------
    def test_scenario_validation_success(self) -> None:
        """Verify valid scenario definitions pass validation."""
        scen = ScenarioDefinition(
            scenario_id="TEST_01",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close R07",
            affected_roads=["R07"],
        )
        is_valid, err = scen.validate(self.test_graph)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_scenario_validation_nonexistent_road(self) -> None:
        """Verify scenario with nonexistent road ID fails validation."""
        scen = ScenarioDefinition(
            scenario_id="TEST_INVALID_ROAD",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close fake road",
            affected_roads=["ROAD_DOES_NOT_EXIST_999"],
        )
        is_valid, err = scen.validate(self.test_graph)
        self.assertFalse(is_valid)
        self.assertIn("does not exist in the road graph", err)

        # Ensure engine returns INVALID_INPUT status without crashing
        report = self.engine.simulate_counterfactual([], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.INVALID_INPUT)
        self.assertIn("rejected", report.uncertainties[0].lower())

    def test_scenario_validation_negative_capacity(self) -> None:
        """Verify negative capacity change is rejected."""
        scen = ScenarioDefinition(
            scenario_id="TEST_NEG_CAP",
            scenario_type=ScenarioType.CAPACITY_REDUCTION,
            description="Negative capacity",
            capacity_changes={"R01": -500.0},
        )
        is_valid, err = scen.validate(self.test_graph)
        self.assertFalse(is_valid)
        self.assertIn("must be non-negative", err)

    def test_scenario_validation_invalid_demand_multiplier(self) -> None:
        """Verify non-positive or invalid demand multiplier is rejected."""
        for bad_mult in [0.0, -1.5, float("nan")]:
            scen = ScenarioDefinition(
                scenario_id="TEST_BAD_MULT",
                scenario_type=ScenarioType.DEMAND_INCREASE,
                description="Invalid multiplier",
                demand_multiplier=bad_mult,
            )
            is_valid, err = scen.validate(self.test_graph)
            self.assertFalse(is_valid)
            self.assertIn("demand_multiplier", err)

    def test_scenario_validation_unsupported_type(self) -> None:
        """Verify unsupported or invalid scenario type fails validation."""
        scen = ScenarioDefinition(
            scenario_id="TEST_UNSUPPORTED",
            scenario_type="NONEXISTENT_TYPE",
            description="Unsupported",
        )
        is_valid, err = scen.validate(self.test_graph)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported or invalid", err)

    # -------------------------------------------------------------------------
    # 2. BASELINE IMMUTABILITY TESTS
    # -------------------------------------------------------------------------
    def test_baseline_immutability(self) -> None:
        """
        Verify that original RoadGraph, NormalizedTrajectory objects,
        candidate routes, and probabilities are 100% untouched after simulation.
        """
        case = self.scenarios_raw["SCN_001_SINGLE_ROAD_CLOSURE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        # Capture deep structural snapshot of baseline inputs
        graph_snapshot_edges = {
            rid: (e.is_closed, e.capacity_vph, e.speed_limit_kmh)
            for rid, e in self.test_graph.edges.items()
        }
        graph_snapshot_adj = copy.deepcopy(self.test_graph.adjacency)

        traj_snapshot_weight = traj.vehicle_weight
        traj_snapshot_routes = [
            (list(r.nodes), r.probability, dict(r.metadata))
            for r in traj.candidate_routes
        ]
        traj_snapshot_meta = dict(traj.metadata)

        # Run counterfactual simulation
        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        # Verify RoadGraph was not mutated
        for rid, (is_closed, cap, spd) in graph_snapshot_edges.items():
            edge = self.test_graph.edges[rid]
            self.assertEqual(edge.is_closed, is_closed, f"Road {rid} is_closed mutated!")
            self.assertEqual(edge.capacity_vph, cap, f"Road {rid} capacity mutated!")
            self.assertEqual(edge.speed_limit_kmh, spd, f"Road {rid} speed mutated!")
        self.assertEqual(self.test_graph.adjacency, graph_snapshot_adj)

        # Verify NormalizedTrajectory object was not mutated
        self.assertEqual(traj.vehicle_weight, traj_snapshot_weight)
        self.assertEqual(traj.metadata, traj_snapshot_meta)
        for idx, r in enumerate(traj.candidate_routes):
            orig_nodes, orig_prob, orig_meta = traj_snapshot_routes[idx]
            self.assertEqual(r.nodes, orig_nodes)
            self.assertEqual(r.probability, orig_prob)
            self.assertEqual(r.metadata, orig_meta)

    # -------------------------------------------------------------------------
    # 3. ROAD CLOSURE & ALTERNATE CORRIDOR TESTS
    # -------------------------------------------------------------------------
    def test_single_road_closure_rerouting_and_zero_flow(self) -> None:
        """
        SCN_001: Close R07.
        Verify:
        - R07 receives 0.0 counterfactual flow.
        - Demand reallocates to surviving route.
        - Alternate corridor absorbs the displaced demand.
        - Original route probabilities remain intact.
        """
        case = self.scenarios_raw["SCN_001_SINGLE_ROAD_CLOSURE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        # Find R07 in baseline and counterfactual
        b_r07 = next(r for r in report.baseline["roads"] if r["road_id"] == "R07")
        cf_r07 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R07")

        self.assertAlmostEqual(b_r07["expected_demand_in_window"], 0.30, places=4)
        self.assertAlmostEqual(cf_r07["expected_demand_in_window"], 0.0, places=4)

        # Surviving route uses R01 and R08
        b_r01 = next(r for r in report.baseline["roads"] if r["road_id"] == "R01")
        cf_r01 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R01")

        self.assertAlmostEqual(b_r01["expected_demand_in_window"], 0.70, places=4)
        self.assertAlmostEqual(cf_r01["expected_demand_in_window"], 1.00, places=4)

        # Impact metrics
        self.assertAlmostEqual(report.impact["displaced_demand"], 0.30, places=4)
        self.assertIn("R01", report.impact["alternate_corridors"])
        self.assertIn("R08", report.impact["alternate_corridors"])
        self.assertIn("R07", report.impact["relieved_roads"])

    def test_multiple_road_closures(self) -> None:
        """
        SCN_004: Close R07, R09, R18 simultaneously.
        Verify all 3 closed roads have 0.0 counterfactual flow and alternatives survive.
        """
        case = self.scenarios_raw["SCN_004_MULTIPLE_ROAD_CLOSURE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        cf_roads = {r["road_id"]: r["expected_demand_in_window"] for r in report.counterfactual["roads"]}
        self.assertEqual(cf_roads["R07"], 0.0)
        self.assertEqual(cf_roads["R09"], 0.0)
        self.assertEqual(cf_roads["R18"], 0.0)

        # Route 3 (R01, R02, R03, R21) survived and receives all 1.0 demand
        self.assertAlmostEqual(cf_roads["R01"], 1.0, places=4)
        self.assertAlmostEqual(cf_roads["R02"], 1.0, places=4)
        self.assertAlmostEqual(cf_roads["R03"], 1.0, places=4)
        self.assertAlmostEqual(cf_roads["R21"], 1.0, places=4)

        self.assertAlmostEqual(report.impact["displaced_demand"], 0.80, places=4)

    # -------------------------------------------------------------------------
    # 4. PROBABILITY SEMANTICS & DEMAND CONSERVATION
    # -------------------------------------------------------------------------
    def test_ambiguous_trajectory_reallocation_semantics(self) -> None:
        """
        SCN_007: Route A=0.50, Route B=0.30, Route C=0.20.
        Close Route A.
        Verify:
        - Route B allocation = 0.30 / 0.50 = 0.60
        - Route C allocation = 0.20 / 0.50 = 0.40
        - Sum of allocations == 1.0 (demand conservation)
        """
        case = self.scenarios_raw["SCN_007_AMBIGUOUS_TRAJECTORY"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        cf_roads = {r["road_id"]: r["expected_demand_in_window"] for r in report.counterfactual["roads"]}
        self.assertEqual(cf_roads["R07"], 0.0)

        # Route B uses R01, R08 -> receives 0.60
        self.assertAlmostEqual(cf_roads["R08"], 0.60, places=4)
        # Route C uses R14, R15, R20 -> receives 0.40
        self.assertAlmostEqual(cf_roads["R14"], 0.40, places=4)
        self.assertAlmostEqual(cf_roads["R15"], 0.40, places=4)
        self.assertAlmostEqual(cf_roads["R20"], 0.40, places=4)

        # Total OD demand is conserved at 1.0
        self.assertAlmostEqual(report.impact["total_counterfactual_demand"], 1.0, places=4)

    def test_low_reliability_not_multiplied_into_demand(self) -> None:
        """
        SCN_008: Low reliability trajectory (0.35).
        Verify physical demand is 1.0, NOT 0.35!
        Reliability must NOT be used as a fractional vehicle multiplier.
        """
        case = self.scenarios_raw["SCN_008_LOW_RELIABILITY"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        cf_roads = {r["road_id"]: r["expected_demand_in_window"] for r in report.counterfactual["roads"]}
        # Surviving route receives full physical demand (1.0)
        self.assertAlmostEqual(cf_roads["R01"], 1.0, places=4)
        self.assertAlmostEqual(cf_roads["R08"], 1.0, places=4)

        # Total demand remains 1.0
        self.assertAlmostEqual(report.impact["total_counterfactual_demand"], 1.0, places=4)

    # -------------------------------------------------------------------------
    # 5. NO FEASIBLE ROUTE & UNROUTABLE DEMAND TESTS
    # -------------------------------------------------------------------------
    def test_no_feasible_route_handling(self) -> None:
        """
        SCN_005: All egress routes from J01 closed (R01, R07, R14).
        Verify:
        - unroutable_demand == 1.0
        - status == NO_FEASIBLE_ROUTE
        - zero fake routes invented
        - counterfactual road demand is 0.0 everywhere
        """
        case = self.scenarios_raw["SCN_005_NO_ALTERNATIVE_ROUTE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.NO_FEASIBLE_ROUTE)
        self.assertAlmostEqual(report.impact["unroutable_demand"], 1.0, places=4)
        self.assertAlmostEqual(report.impact["total_counterfactual_demand"], 0.0, places=4)

        # Check OD impact
        od_imp = report.impact["od_impacts"][0]
        self.assertEqual(od_imp["origin"], "J01")
        self.assertEqual(od_imp["destination"], "J12")
        self.assertAlmostEqual(od_imp["unroutable_demand"], 1.0, places=4)
        self.assertAlmostEqual(od_imp["counterfactual_demand"], 0.0, places=4)

    # -------------------------------------------------------------------------
    # 6. CAPACITY CHANGES TESTS
    # -------------------------------------------------------------------------
    def test_capacity_reduction(self) -> None:
        """
        SCN_002: Reduce R01 capacity by 50% (1600 -> 800 vph).
        Verify scenario capacity is 800 vph, utilization doubles, and no fake rerouting occurs.
        """
        case = self.scenarios_raw["SCN_002_CAPACITY_REDUCTION"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        b_r01 = next(r for r in report.baseline["roads"] if r["road_id"] == "R01")
        cf_r01 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R01")

        self.assertEqual(b_r01["capacity_vph"], 1600.0)
        self.assertEqual(cf_r01["capacity_vph"], 800.0)

        # Demand did NOT change because road is not closed
        self.assertAlmostEqual(b_r01["expected_demand_in_window"], cf_r01["expected_demand_in_window"], places=4)

        # Utilization ratio doubled
        self.assertAlmostEqual(cf_r01["utilization_ratio"], b_r01["utilization_ratio"] * 2.0, places=4)

    def test_capacity_increase(self) -> None:
        """
        SCN_003: Increase R07 capacity (1400 -> 2800 vph).
        Verify scenario capacity is 2800 vph and utilization drops by half.
        """
        case = self.scenarios_raw["SCN_003_CAPACITY_INCREASE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        b_r07 = next(r for r in report.baseline["roads"] if r["road_id"] == "R07")
        cf_r07 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R07")

        self.assertEqual(b_r07["capacity_vph"], 1400.0)
        self.assertEqual(cf_r07["capacity_vph"], 2800.0)
        self.assertAlmostEqual(cf_r07["utilization_ratio"], b_r07["utilization_ratio"] * 0.5, places=4)

    # -------------------------------------------------------------------------
    # 7. DEMAND CHANGES TESTS
    # -------------------------------------------------------------------------
    def test_demand_multiplier_increase_and_decrease(self) -> None:
        """
        SCN_006: Demand multiplier = 1.25 (+25%).
        Also test demand multiplier = 0.75 (-25%).
        """
        case = self.scenarios_raw["SCN_006_HIGH_DEMAND"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        # Test +25%
        scen_inc = ScenarioDefinition.from_dict(case)
        report_inc = self.engine.simulate_counterfactual([traj], self.test_graph, scen_inc)
        self.assertEqual(report_inc.status, ScenarioStatus.SUCCESS)

        b_r01 = next(r for r in report_inc.baseline["roads"] if r["road_id"] == "R01")
        cf_r01 = next(r for r in report_inc.counterfactual["roads"] if r["road_id"] == "R01")

        self.assertAlmostEqual(b_r01["expected_demand_in_window"], 2.0, places=4)
        self.assertAlmostEqual(cf_r01["expected_demand_in_window"], 2.5, places=4)
        self.assertAlmostEqual(cf_r01["expected_demand_in_window"], b_r01["expected_demand_in_window"] * 1.25, places=4)

        # Test -25%
        scen_dec = ScenarioDefinition(
            scenario_id="TEST_DEMAND_DEC",
            scenario_type=ScenarioType.DEMAND_DECREASE,
            description="Hypothetical 25% demand drop",
            demand_multiplier=0.75,
        )
        report_dec = self.engine.simulate_counterfactual([traj], self.test_graph, scen_dec)
        cf_r01_dec = next(r for r in report_dec.counterfactual["roads"] if r["road_id"] == "R01")
        self.assertAlmostEqual(cf_r01_dec["expected_demand_in_window"], 1.5, places=4)

    # -------------------------------------------------------------------------
    # 8. ROAD RECOVERY TESTS
    # -------------------------------------------------------------------------
    def test_road_recovery_scenario(self) -> None:
        """
        SCN_009: Road R07 is initially closed in baseline.
        Scenario: ROAD_RECOVERY on R07.
        Verify:
        - R07 is closed in baseline.
        - R07 is restored and receives flow in counterfactual.
        """
        case = self.scenarios_raw["SCN_009_ROAD_RECOVERY"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        # Create baseline graph where R07 is closed
        base_graph = copy.deepcopy(self.test_graph)
        base_graph.close_road("R07")
        self.assertTrue(base_graph.is_road_closed("R07"))

        report = self.engine.simulate_counterfactual([traj], base_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        # Baseline R07 has 0 flow
        b_r07 = next(r for r in report.baseline["roads"] if r["road_id"] == "R07")
        self.assertEqual(b_r07["expected_demand_in_window"], 0.0)

        # Counterfactual R07 is restored and receives 0.50 flow
        cf_r07 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R07")
        self.assertAlmostEqual(cf_r07["expected_demand_in_window"], 0.50, places=4)

        # Base graph still has R07 closed (immutability)
        self.assertTrue(base_graph.is_road_closed("R07"))

    # -------------------------------------------------------------------------
    # 9. TEMPORAL INTEGRITY & UNCALIBRATED TIME WINDOWS
    # -------------------------------------------------------------------------
    def test_missing_time_window_remains_uncalibrated(self) -> None:
        """
        Verify that a trajectory without time windows produces:
        - is_hourly_rate_valid = False
        - expected_demand_vph = None
        - utilization_ratio = None
        - RoadStatus = UNCALIBRATED
        Zero fake travel time or hourly rates fabricated!
        """
        traj = NormalizedTrajectory(
            track_id="VEH_NO_TIME",
            origin_node="J01",
            destination_node="J02",
            candidate_routes=[
                NormalizedCandidateRoute(nodes=["J01", "J02"], probability=1.0, metadata={"edges": ["R01"]})
            ],
            vehicle_weight=1.0,
            time_window_start=None,
            time_window_end=None,
        )
        scen = ScenarioDefinition(
            scenario_id="TEST_TIME_SAFE",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close R07",
            affected_roads=["R07"],
        )

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        cf_r01 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R01")

        self.assertFalse(cf_r01["is_hourly_rate_valid"])
        self.assertIsNone(cf_r01["expected_demand_vph"])
        self.assertIsNone(cf_r01["utilization_ratio"])
        self.assertEqual(cf_r01["status"], RoadStatus.UNCALIBRATED.value)

    # -------------------------------------------------------------------------
    # 10. EDGE CASES & SAFE DIVISION
    # -------------------------------------------------------------------------
    def test_empty_trajectory_list(self) -> None:
        """Verify empty trajectory list runs safely with zero demand."""
        scen = ScenarioDefinition(
            scenario_id="TEST_EMPTY",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close R07 with no traffic",
            affected_roads=["R07"],
        )
        report = self.engine.simulate_counterfactual([], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)
        self.assertEqual(report.impact["total_counterfactual_demand"], 0.0)
        self.assertEqual(report.impact["displaced_demand"], 0.0)

    def test_safe_percent_division_by_zero(self) -> None:
        """Verify safe handling of 0 baseline demand (never fabricates inf or div-by-zero)."""
        # Road with 0 baseline demand receives demand in counterfactual
        analyzer = ImpactAnalyzer(self.test_graph)
        scen = ScenarioDefinition(
            scenario_id="TEST_DIV_ZERO",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Div zero test",
            affected_roads=["R07"],
        )

        from schemas.mobility_schema import RoadFlowMetric, ODMatrix
        b_road = RoadFlowMetric(road_id="R01", from_node="J01", to_node="J02", expected_demand_in_window=0.0)
        cf_road = RoadFlowMetric(road_id="R01", from_node="J01", to_node="J02", expected_demand_in_window=1.5)

        summary = analyzer.analyze(
            scenario=scen,
            baseline_roads=[b_road],
            counterfactual_roads=[cf_road],
            baseline_od=ODMatrix(pairs=[]),
            counterfactual_od=ODMatrix(pairs=[]),
        )

        r01_impact = summary["road_impacts"][0]
        self.assertIsNone(r01_impact["demand_delta_percent"])
        self.assertEqual(r01_impact["demand_delta"], 1.5)

    def test_scenario_10_invalid_scenario_direct_load(self) -> None:
        """Verify SCN_010 from JSON file loads and fails validation cleanly."""
        case = self.scenarios_raw["SCN_010_INVALID_SCENARIO"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.INVALID_INPUT)
        self.assertEqual(report.impact["unroutable_demand"], 0.0)
        self.assertIn("rejected", report.uncertainties[0].lower())

    def test_alternate_route_discovery_when_baseline_has_no_surviving_route(self) -> None:
        """
        Verify that when a trajectory's baseline route is closed,
        the engine calls find_candidate_paths() to discover surviving alternatives
        on the counterfactual network rather than giving up immediately.
        """
        # Baseline trajectory has ONLY one candidate route: J01 -> J04 (uses R07)
        traj = NormalizedTrajectory(
            track_id="VEH_SINGLE_ROUTE",
            origin_node="J01",
            destination_node="J04",
            candidate_routes=[
                NormalizedCandidateRoute(nodes=["J01", "J04"], probability=1.0, metadata={"edges": ["R07"]})
            ],
            vehicle_weight=1.0,
            time_window_start=1000.0,
            time_window_end=1100.0,
        )
        # Close R07. Alternative route exists: J01 -> J02 -> J05 -> J08 -> J04, or J01 -> J13 -> J07 -> J04 (R14, R15, R20)
        scen = ScenarioDefinition(
            scenario_id="TEST_REROUTE_DISCOVERY",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close R07; alternative via J13/J07 exists",
            affected_roads=["R07"],
        )

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertEqual(report.status, ScenarioStatus.SUCCESS)

        # Closed road has 0 flow
        cf_r07 = next(r for r in report.counterfactual["roads"] if r["road_id"] == "R07")
        self.assertEqual(cf_r07["expected_demand_in_window"], 0.0)

        # Displaced demand is 1.0
        self.assertAlmostEqual(report.impact["displaced_demand"], 1.0, places=4)
        self.assertAlmostEqual(report.impact["total_counterfactual_demand"], 1.0, places=4)

        # An alternate corridor absorbed the demand
        self.assertTrue(len(report.impact["alternate_corridors"]) > 0)

    def test_bottleneck_spillover_detection(self) -> None:
        """
        Verify that a newly congested road sharing a junction with an affected road
        is flagged as bottleneck spillover.
        """
        analyzer = ImpactAnalyzer(self.test_graph)
        scen = ScenarioDefinition(
            scenario_id="TEST_SPILLOVER",
            scenario_type=ScenarioType.ROAD_CLOSURE,
            description="Close R07",
            affected_roads=["R07"],  # R07 connects J01 -> J04
        )

        from schemas.mobility_schema import RoadFlowMetric, RoadStatus, ODMatrix
        # R11 connects J04 -> J08 (shares node J04 with intervened road R07)
        b_road_r11 = RoadFlowMetric(
            road_id="R11", from_node="J04", to_node="J08",
            capacity_vph=1200.0, expected_demand_in_window=100.0,
            expected_demand_vph=600.0, utilization_ratio=0.50,
            is_hourly_rate_valid=True, status=RoadStatus.MODERATE
        )
        # In counterfactual, R11 receives heavy rerouted traffic and becomes HIGH
        cf_road_r11 = RoadFlowMetric(
            road_id="R11", from_node="J04", to_node="J08",
            capacity_vph=1200.0, expected_demand_in_window=200.0,
            expected_demand_vph=1050.0, utilization_ratio=0.875,
            is_hourly_rate_valid=True, status=RoadStatus.HIGH
        )

        summary = analyzer.analyze(
            scenario=scen,
            baseline_roads=[b_road_r11],
            counterfactual_roads=[cf_road_r11],
            baseline_od=ODMatrix(pairs=[]),
            counterfactual_od=ODMatrix(pairs=[]),
        )

        self.assertIn("R11", summary["new_bottlenecks"])
        self.assertIn("R11", summary["bottleneck_spillover"])

    def test_scenario_assumptions_and_framing(self) -> None:
        """Verify scenario reports contain clear hypothetical assumptions framing."""
        case = self.scenarios_raw["SCN_001_SINGLE_ROAD_CLOSURE"]
        scen = ScenarioDefinition.from_dict(case)
        traj = NormalizedTrajectory.from_dict(case["trajectory"])

        report = self.engine.simulate_counterfactual([traj], self.test_graph, scen)
        self.assertTrue(any("plausible network consequences" in a for a in report.assumptions))
        self.assertTrue(any("hypothetical" in a.lower() for a in report.assumptions))


if __name__ == "__main__":
    unittest.main()

