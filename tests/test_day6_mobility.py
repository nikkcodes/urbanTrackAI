"""
Day 6 Test Suite: City Mobility Graph Integration & Trajectory → Traffic Flow.

Tests:
- Route demand formula: vehicle_weight * route_probability
- Demand conservation invariant: sum(route_demand) == vehicle_weight
- Distinction between route demand conservation and road demand sum
- Shared road demand aggregation
- Multiple vehicle aggregation
- Vehicle weight scaling (PCU heavy vehicle)
- Origin-Destination (OD) matrix independence from route probabilities
- Closed road handling and exclusion
- Reliability & uncertainty independence (physical demand is NOT multiplied by reliability)
- Safe capacity division and status classification
- Pure-Python network structural centrality (degree and Brandes betweenness)
- Traffic priority road ranking
- Graceful validation error handling on malformed inputs
- Time window propagation
- Full CityMobilityReport serialization
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from inference.road_graph import RoadGraph
from schemas.normalized_trajectory_schema import (
    InvalidRouteError,
    NormalizedCandidateRoute,
    NormalizedTrajectory,
    ProbabilityValidationError,
)
from schemas.mobility_schema import (
    CityMobilityReport,
    ODMatrix,
    ODPairDemand,
    RoadFlowMetric,
    RoadPriority,
    RoadStatus,
)
from mobility.flow_engine import MobilityFlowEngine
from mobility.network_analytics import NetworkCentralityAnalyzer, PriorityRoadRanker
from mobility.mobility_engine import CityMobilityEngine


class TestDay6MobilityAnalytics(unittest.TestCase):
    """Automated unit and integration test suite for Day 6."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.network_path = Path("data/synthetic/city_network.json")
        cls.graph = RoadGraph.from_json_file(cls.network_path)
        cls.scenarios_path = Path("data/synthetic/day6_mobility_scenarios.json")
        with open(cls.scenarios_path, "r", encoding="utf-8") as f:
            cls.scenarios = json.load(f)["scenarios"]

    def setUp(self) -> None:
        # Re-load graph to ensure clean state per test (especially if closed road tests modify edges)
        self.graph = RoadGraph.from_json_file(self.network_path)
        self.flow_engine = MobilityFlowEngine(self.graph)
        self.mobility_engine = CityMobilityEngine(self.graph)

    # =========================================================================
    # CASE 1: SINGLE CLEAR ROUTE
    # =========================================================================
    def test_case1_single_clear_route(self) -> None:
        raw = self.scenarios["case1_single_clear_route"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows([traj])

        self.assertEqual(len(issues), 0)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_conserved"])

        metric_map = {m.road_id: m for m in road_metrics}
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 1.0, places=4)
        self.assertAlmostEqual(metric_map["R02"].expected_demand, 1.0, places=4)
        self.assertAlmostEqual(metric_map["R03"].expected_demand, 0.0, places=4)

        # OD demand
        self.assertAlmostEqual(od_matrix.get_demand("J01", "J03"), 1.0, places=4)
        self.assertEqual(od_matrix.total_demand, 1.0)

    # =========================================================================
    # CASE 2: TWO ROUTES
    # =========================================================================
    def test_case2_two_routes(self) -> None:
        raw = self.scenarios["case2_two_routes"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows([traj])

        self.assertEqual(len(issues), 0)
        metric_map = {m.road_id: m for m in road_metrics}

        # Route A (0.70): R01, R08
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 0.70, places=4)
        self.assertAlmostEqual(metric_map["R08"].expected_demand, 0.70, places=4)
        # Route B (0.30): R07, R18
        self.assertAlmostEqual(metric_map["R07"].expected_demand, 0.30, places=4)
        self.assertAlmostEqual(metric_map["R18"].expected_demand, 0.30, places=4)

        # Invariant check
        self.assertTrue(records[0]["is_conserved"])
        self.assertAlmostEqual(records[0]["sum_route_demand"], 1.0, places=4)

    # =========================================================================
    # CASE 3: SHARED ROAD
    # =========================================================================
    def test_case3_shared_road(self) -> None:
        raw = self.scenarios["case3_shared_road"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows([traj])

        metric_map = {m.road_id: m for m in road_metrics}
        # Shared R01: 0.60 + 0.40 = 1.00
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 1.00, places=4)
        self.assertAlmostEqual(metric_map["R08"].expected_demand, 0.60, places=4)
        self.assertAlmostEqual(metric_map["R02"].expected_demand, 0.40, places=4)

        # Contributing trajectories count on R01 should be 1
        self.assertEqual(metric_map["R01"].contributing_trajectories_count, 1)

    # =========================================================================
    # CASE 4: HEAVY VEHICLE WEIGHT SCALING
    # =========================================================================
    def test_case4_heavy_vehicle(self) -> None:
        raw = self.scenarios["case4_heavy_vehicle"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows([traj])

        metric_map = {m.road_id: m for m in road_metrics}
        # Demand scaled by 2.5
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 2.5, places=4)
        self.assertAlmostEqual(metric_map["R02"].expected_demand, 2.5, places=4)

        # OD demand also scaled by 2.5
        self.assertAlmostEqual(od_matrix.get_demand("J01", "J03"), 2.5, places=4)

        # Conservation holds: sum_route_demand == 2.5
        self.assertTrue(records[0]["is_conserved"])
        self.assertAlmostEqual(records[0]["sum_route_demand"], 2.5, places=4)

    # =========================================================================
    # CASE 5: MULTIPLE VEHICLES AGGREGATION
    # =========================================================================
    def test_case5_multiple_vehicles(self) -> None:
        raw_list = self.scenarios["case5_multiple_vehicles"]["trajectories"]
        trajs = [NormalizedTrajectory.from_dict(r) for r in raw_list]

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows(trajs)

        metric_map = {m.road_id: m for m in road_metrics}
        # R01 used by VEH_05_A (1.0) and VEH_05_B (2.0) -> 3.0
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 3.0, places=4)
        self.assertEqual(metric_map["R01"].contributing_trajectories_count, 2)
        self.assertIn("VEH_05_A", metric_map["R01"].contributing_identities)
        self.assertIn("VEH_05_B", metric_map["R01"].contributing_identities)

        # R08 used by VEH_05_B (2.0) and VEH_05_C (1.5) -> 3.5
        self.assertAlmostEqual(metric_map["R08"].expected_demand, 3.5, places=4)

        # R02 used only by VEH_05_A (1.0) -> 1.0
        self.assertAlmostEqual(metric_map["R02"].expected_demand, 1.0, places=4)

    # =========================================================================
    # CASE 6: DIFFERENT OD PAIRS
    # =========================================================================
    def test_case6_different_od_pairs(self) -> None:
        raw_list = self.scenarios["case6_different_od_pairs"]["trajectories"]
        trajs = [NormalizedTrajectory.from_dict(r) for r in raw_list]

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows(trajs)

        # OD Matrix must isolate origin-destination pairs
        self.assertAlmostEqual(od_matrix.get_demand("J01", "J03"), 3.0, places=4)
        self.assertAlmostEqual(od_matrix.get_demand("J04", "J08"), 1.5, places=4)
        self.assertAlmostEqual(od_matrix.get_demand("J01", "J08"), 0.0, places=4)
        self.assertAlmostEqual(od_matrix.total_demand, 4.5, places=4)

    # =========================================================================
    # CASE 7: CLOSED ROADS HANDLING
    # =========================================================================
    def test_case7_closed_road_exclusion(self) -> None:
        raw = self.scenarios["case7_closed_road"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        # Close R02 in graph
        self.graph.close_road("R02")
        self.assertTrue(self.graph.edges["R02"].is_closed)

        road_metrics, od_matrix, issues, records = self.flow_engine.aggregate_flows([traj])

        metric_map = {m.road_id: m for m in road_metrics}
        # Closed road must not receive valid demand
        self.assertAlmostEqual(metric_map["R02"].expected_demand, 0.0, places=4)
        self.assertAlmostEqual(metric_map["R01"].expected_demand, 0.0, places=4)

        # Issue should be recorded
        closed_issues = [i for i in issues if i.issue_type in ("closed_road_traversed", "no_feasible_routes")]
        self.assertTrue(len(closed_issues) > 0)

    # =========================================================================
    # CASE 8 & 9: AMBIGUOUS TRAJECTORY & RELIABILITY INDEPENDENCE
    # =========================================================================
    def test_case8_and_case9_reliability_independence(self) -> None:
        raw8 = self.scenarios["case8_ambiguous_trajectory"]["trajectory"]
        raw9 = self.scenarios["case9_low_reliability"]["trajectory"]

        traj8 = NormalizedTrajectory.from_dict(raw8)
        traj9 = NormalizedTrajectory.from_dict(raw9)

        # Run Case 8 (high reliability 0.85)
        metrics8, _, _, _ = self.flow_engine.aggregate_flows([traj8])
        # Run Case 9 (low reliability 0.20)
        metrics9, _, _, _ = self.flow_engine.aggregate_flows([traj9])

        m8_map = {m.road_id: m for m in metrics8}
        m9_map = {m.road_id: m for m in metrics9}

        # CRITICAL TEST: Physical expected demand must be EXACTLY identical!
        for rid in ("R01", "R08", "R07", "R18"):
            self.assertAlmostEqual(m8_map[rid].expected_demand, 0.50, places=4)
            self.assertAlmostEqual(m9_map[rid].expected_demand, 0.50, places=4)
            self.assertAlmostEqual(m8_map[rid].expected_demand, m9_map[rid].expected_demand, places=6)

        # However, reliability metadata MUST reflect the differing evidence quality
        self.assertAlmostEqual(m8_map["R01"].reliability_summary["mean_reliability"], 0.85, places=2)
        self.assertAlmostEqual(m9_map["R01"].reliability_summary["mean_reliability"], 0.20, places=2)

    # =========================================================================
    # CASE 10: INVALID INPUTS
    # =========================================================================
    def test_case10_invalid_inputs_resilience(self) -> None:
        raw_invalids = self.scenarios["case10_invalid_inputs"]["invalid_trajectories"]

        # 1. Negative vehicle weight
        with self.assertRaises(ValueError):
            NormalizedTrajectory.from_dict(raw_invalids[0])

        # 2. Unknown node in graph
        traj_unknown_node = NormalizedTrajectory.from_dict(raw_invalids[1])
        valid, issues, _ = self.flow_engine.validate_trajectory(traj_unknown_node)
        self.assertFalse(valid)
        self.assertTrue(any(i.issue_type == "unknown_origin_node" for i in issues))

        # 3. Probability sum != 1.0 beyond recovery tolerance
        with self.assertRaises(ProbabilityValidationError):
            NormalizedTrajectory.from_dict(raw_invalids[2])

        # 4. Empty candidate routes
        with self.assertRaises(ValueError):
            NormalizedTrajectory.from_dict(raw_invalids[3])

        # 5. Empty trajectory list does not crash engine
        empty_metrics, empty_od, empty_issues, empty_records = self.flow_engine.aggregate_flows([])
        self.assertEqual(len(empty_metrics), len(self.graph.edges))
        self.assertEqual(empty_od.total_demand, 0.0)

    # =========================================================================
    # PART 16 & 17: DEMAND CONSERVATION INVARIANTS
    # =========================================================================
    def test_demand_conservation_invariant(self) -> None:
        """
        Verify that for any trajectory:
        sum(route_demand across candidate routes) == vehicle_weight
        while sum(road_demand across network) != vehicle_weight.
        """
        weight = 2.5
        routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=0.40, metadata={"edges": ["R01", "R02"]}),
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.35, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.25, metadata={"edges": ["R07", "R18"]}),
        ]
        # Verify endpoint validation raises InvalidRouteError when route end != destination
        with self.assertRaises(InvalidRouteError):
            NormalizedTrajectory(
                track_id="VEH_CONSERVATION_TEST",
                origin_node="J01",
                destination_node="J03",  # Route 2 and 3 destination mismatch
                candidate_routes=routes,
                vehicle_weight=weight,
            )

        # Now fix destination to J05 and make route 1 reach J05 via J03->J06->J08->J05
        corrected_routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.40, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.35, metadata={"edges": ["R07", "R18"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J08", "J05"], probability=0.25, metadata={"edges": ["R07", "R11", "R26"]}),
        ]
        traj_valid = NormalizedTrajectory(
            track_id="VEH_CONSERVATION_OK",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=corrected_routes,
            vehicle_weight=weight,
        )

        road_metrics, _, _, records = self.flow_engine.aggregate_flows([traj_valid])

        # 1. Route demand conservation
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_conserved"])
        self.assertAlmostEqual(records[0]["sum_route_demand"], weight, places=6)

        # 2. Road demand sum across all road segments
        total_road_demand = sum(m.expected_demand for m in road_metrics)
        # Each route has 2 or 3 edges, so total road demand must exceed vehicle_weight
        self.assertGreater(total_road_demand, weight)

    # =========================================================================
    # PART 4: ROAD UTILIZATION AND STATUS CLASSIFICATION
    # =========================================================================
    def test_utilization_classification_and_safe_division(self) -> None:
        self.assertEqual(RoadStatus.classify(0.15), RoadStatus.NORMAL)
        self.assertEqual(RoadStatus.classify(0.49), RoadStatus.NORMAL)
        self.assertEqual(RoadStatus.classify(0.50), RoadStatus.MODERATE)
        self.assertEqual(RoadStatus.classify(0.79), RoadStatus.MODERATE)
        self.assertEqual(RoadStatus.classify(0.80), RoadStatus.HIGH)
        self.assertEqual(RoadStatus.classify(1.00), RoadStatus.HIGH)
        self.assertEqual(RoadStatus.classify(1.05), RoadStatus.OVER_CAPACITY)

        # Test safe zero/negative capacity handling
        metric = RoadFlowMetric(
            road_id="R_TEST",
            from_node="J01",
            to_node="J02",
            expected_demand=100.0,
            capacity_vph=0.0,
            utilization_ratio=0.0,
        )
        self.assertEqual(metric.utilization_ratio, 0.0)

    # =========================================================================
    # PART 8: PURE PYTHON NETWORK CENTRALITY
    # =========================================================================
    def test_pure_python_centrality(self) -> None:
        analyzer = NetworkCentralityAnalyzer(self.graph)

        # 1. Degree metrics
        degrees = analyzer.compute_degree_metrics()
        self.assertIn("J01", degrees)
        in_deg, out_deg, tot_deg = degrees["J01"]
        self.assertEqual(tot_deg, in_deg + out_deg)
        self.assertGreater(out_deg, 0)

        # 2. Betweenness centrality
        cb = analyzer.compute_edge_betweenness(normalized=True)
        self.assertEqual(len(cb), len(self.graph.edges))
        for rid, val in cb.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        # Central roads like R01, R08, R09 should have non-zero betweenness
        centrality_map = analyzer.analyze_network()
        self.assertIn("R01", centrality_map)
        self.assertEqual(centrality_map["R01"].metric_type, "network_structural_importance")

    # =========================================================================
    # PART 9: PRIORITY ROAD RANKING
    # =========================================================================
    def test_priority_road_ranking(self) -> None:
        raw_list = self.scenarios["case5_multiple_vehicles"]["trajectories"]
        trajs = [NormalizedTrajectory.from_dict(r) for r in raw_list]

        report = self.mobility_engine.process_trajectories(trajs, top_priority_count=5)

        self.assertIsInstance(report, CityMobilityReport)
        self.assertEqual(len(report.priority_roads), 5)

        top_road = report.priority_roads[0]
        # Road with highest utilization/demand should be top priority
        self.assertIn(top_road.road_id, ("R08", "R01"))
        self.assertGreater(top_road.priority_score, 0.0)
        self.assertEqual(top_road.rank, 1)

    # =========================================================================
    # PART 18 & 19: CITY MOBILITY REPORT & TIME WINDOWS
    # =========================================================================
    def test_city_mobility_report_serialization(self) -> None:
        raw = self.scenarios["case1_single_clear_route"]["trajectory"]
        traj = NormalizedTrajectory.from_dict(raw)

        report = self.mobility_engine.process_trajectories(
            [traj],
            time_window={"start": 1000.0, "end": 1450.0},
        )

        d = report.to_dict()
        self.assertIn("time_window", d)
        self.assertEqual(d["time_window"]["start"], 1000.0)
        self.assertEqual(d["time_window"]["end"], 1450.0)
        self.assertIn("road_metrics", d)
        self.assertIn("od_demand", d)
        self.assertIn("priority_roads", d)
        self.assertIn("network_centrality", d)
        self.assertIn("summary", d)

        # Verify JSON serialization round-trip
        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["summary"]["valid_trajectories_count"], 1)
        self.assertTrue(parsed["summary"]["demand_conservation_verified"])

    # =========================================================================
    # CORRECTIVE AUDIT TESTS (SECTION 16)
    # =========================================================================
    def test_audit_01_demand_conservation_multiroute(self) -> None:
        """Audit 1: Verify demand conservation invariant across multi-route trajectories."""
        weight = 3.5
        routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.40, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.35, metadata={"edges": ["R07", "R18"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J08", "J05"], probability=0.25, metadata={"edges": ["R07", "R11", "R26"]}),
        ]
        traj = NormalizedTrajectory(
            track_id="AUDIT_CONSERVATION_01",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes,
            vehicle_weight=weight,
        )
        road_metrics, _, _, records = self.flow_engine.aggregate_flows([traj])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["is_conserved"])
        self.assertAlmostEqual(records[0]["sum_route_demand"], weight, places=6)

    def test_audit_02_shared_edge_aggregation_no_double_counting(self) -> None:
        """Audit 2: Verify shared road aggregation does not double count physical vehicles."""
        weight = 1.0
        routes_shared = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.60, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J02", "J03", "J06", "J08", "J05"], probability=0.40, metadata={"edges": ["R01", "R02", "R03", "R21", "R26"]}),
        ]
        traj = NormalizedTrajectory(
            track_id="AUDIT_SHARED_02",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes_shared,
            vehicle_weight=weight,
        )
        road_metrics, _, _, _ = self.flow_engine.aggregate_flows([traj])
        metric_map = {m.road_id: m for m in road_metrics}
        # R01 demand should equal 0.60 + 0.40 = 1.00
        self.assertAlmostEqual(metric_map["R01"].expected_demand_in_window, 1.0, places=4)
        # Contributing trajectory count should strictly be 1, NOT 2
        self.assertEqual(metric_map["R01"].contributing_trajectories_count, 1)
        self.assertEqual(metric_map["R01"].contributing_identities, ["AUDIT_SHARED_02"])

    def test_audit_03_reliability_independence_controlled(self) -> None:
        """Audit 3: Controlled comparison proving reliability does not change physical demand."""
        routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.70, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.30, metadata={"edges": ["R07", "R18"]}),
        ]
        traj_high_rel = NormalizedTrajectory(
            track_id="AUDIT_HIGH_REL",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes,
            vehicle_weight=1.0,
            metadata={"trajectory_reliability": 0.98, "trajectory_uncertainty": 0.02},
        )
        traj_low_rel = NormalizedTrajectory(
            track_id="AUDIT_LOW_REL",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes,
            vehicle_weight=1.0,
            metadata={"trajectory_reliability": 0.12, "trajectory_uncertainty": 0.88},
        )
        metrics_high, _, _, _ = self.flow_engine.aggregate_flows([traj_high_rel])
        metrics_low, _, _, _ = self.flow_engine.aggregate_flows([traj_low_rel])

        map_high = {m.road_id: m for m in metrics_high}
        map_low = {m.road_id: m for m in metrics_low}

        for rid in ("R01", "R08", "R07", "R18"):
            self.assertEqual(map_high[rid].expected_demand_in_window, map_low[rid].expected_demand_in_window)

        self.assertAlmostEqual(map_high["R01"].reliability_summary["mean_reliability"], 0.98, places=2)
        self.assertAlmostEqual(map_low["R01"].reliability_summary["mean_reliability"], 0.12, places=2)

    def test_audit_04_od_probability_independence(self) -> None:
        """Audit 4: OD demand strictly accumulates vehicle_weight unaffected by route splits."""
        t1 = NormalizedTrajectory(
            track_id="OD_T1",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0, metadata={"edges": ["R01", "R08"]})],
            vehicle_weight=2.0,
        )
        t2 = NormalizedTrajectory(
            track_id="OD_T2",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[
                NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.80, metadata={"edges": ["R01", "R08"]}),
                NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.20, metadata={"edges": ["R07", "R18"]}),
            ],
            vehicle_weight=3.0,
        )
        _, od_matrix, _, _ = self.flow_engine.aggregate_flows([t1, t2])
        self.assertAlmostEqual(od_matrix.get_demand("J01", "J05"), 5.0, places=4)
        self.assertEqual(od_matrix.pairs[0].contributing_trajectories_count, 2)

    def test_audit_05_closed_route_renormalization_case_b(self) -> None:
        """Audit 5: Case B - One route closed; original probability preserved, flow re-normalized."""
        routes_j05 = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.70, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.30, metadata={"edges": ["R07", "R18"]}),
        ]
        traj = NormalizedTrajectory(
            track_id="AUDIT_CLOSED_B",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes_j05,
            vehicle_weight=1.0,
        )
        # Close R08
        self.graph.close_road("R08")
        self.assertTrue(self.graph.edges["R08"].is_closed)

        valid, issues, allocated_routes = self.flow_engine.validate_and_allocate_routes(traj)
        self.assertTrue(valid)
        self.assertEqual(len(allocated_routes), 1)

        ar = allocated_routes[0]
        # Original probability is untouched at 0.30
        self.assertAlmostEqual(ar.original_route_probability, 0.30, places=4)
        # Flow allocation probability is re-normalized to 1.00 for flow assignment
        self.assertAlmostEqual(ar.flow_allocation_probability, 1.00, places=4)
        self.assertTrue(ar.network_constraint_applied)

        road_metrics, _, _, records = self.flow_engine.aggregate_flows([traj])
        metric_map = {m.road_id: m for m in road_metrics}

        # Closed road R08 receives 0.0 flow
        self.assertAlmostEqual(metric_map["R08"].expected_demand_in_window, 0.0, places=4)
        # Feasible Route B edges (R07, R18) receive full 1.0 flow
        self.assertAlmostEqual(metric_map["R07"].expected_demand_in_window, 1.0, places=4)
        self.assertAlmostEqual(metric_map["R18"].expected_demand_in_window, 1.0, places=4)

    def test_audit_06_all_routes_closed_case_c(self) -> None:
        """Audit 6: Case C - All candidate routes closed; zero flow allocated, explicit issue recorded."""
        routes = [
            NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=0.70, metadata={"edges": ["R01", "R08"]}),
            NormalizedCandidateRoute(nodes=["J01", "J04", "J05"], probability=0.30, metadata={"edges": ["R07", "R18"]}),
        ]
        traj = NormalizedTrajectory(
            track_id="AUDIT_ALL_CLOSED_C",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=routes,
            vehicle_weight=1.0,
        )
        # Close both R01 and R07
        self.graph.close_road("R01")
        self.graph.close_road("R07")

        valid, issues, allocated_routes = self.flow_engine.validate_and_allocate_routes(traj)
        self.assertFalse(valid)
        self.assertEqual(len(allocated_routes), 0)
        self.assertTrue(any(i.issue_type == "all_candidate_routes_closed" for i in issues))

        road_metrics, _, _, records = self.flow_engine.aggregate_flows([traj])
        metric_map = {m.road_id: m for m in road_metrics}
        self.assertAlmostEqual(metric_map["R01"].expected_demand_in_window, 0.0, places=4)
        self.assertAlmostEqual(metric_map["R07"].expected_demand_in_window, 0.0, places=4)
        self.assertFalse(records[0]["is_conserved"])

    def test_audit_07_time_window_hourly_conversion(self) -> None:
        """Audit 7: Hourly rate conversion and dimensional utilization ratio."""
        traj = NormalizedTrajectory(
            track_id="AUDIT_HOURLY",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0, metadata={"edges": ["R01", "R08"]})],
            vehicle_weight=2.0,
            time_window_start=1000.0,
            time_window_end=1450.0,  # 450 seconds = 0.125 hours
        )
        road_metrics, _, _, _ = self.flow_engine.aggregate_flows([traj])
        m_r01 = next(m for m in road_metrics if m.road_id == "R01")

        self.assertTrue(m_r01.is_hourly_rate_valid)
        self.assertEqual(m_r01.duration_seconds, 450.0)
        # expected_demand_in_window = 2.0
        self.assertAlmostEqual(m_r01.expected_demand_in_window, 2.0, places=4)
        # expected_demand_vph = 2.0 / (450 / 3600) = 2.0 / 0.125 = 16.0 vph
        self.assertAlmostEqual(m_r01.expected_demand_vph, 16.0, places=4)
        # capacity of R01 = 1600.0 vph
        # utilization_ratio = 16.0 / 1600.0 = 0.0100
        self.assertAlmostEqual(m_r01.utilization_ratio, 0.0100, places=4)
        self.assertEqual(m_r01.status, RoadStatus.NORMAL)

    def test_audit_08_zero_and_negative_duration_window(self) -> None:
        """Audit 8: Zero, negative, or missing duration preserves window demand without false vph."""
        traj_zero = NormalizedTrajectory(
            track_id="AUDIT_ZERO_DUR",
            origin_node="J01",
            destination_node="J05",
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J05"], probability=1.0, metadata={"edges": ["R01", "R08"]})],
            vehicle_weight=1.0,
            time_window_start=1000.0,
            time_window_end=1000.0,  # 0s duration
        )
        road_metrics, _, issues, _ = self.flow_engine.aggregate_flows([traj_zero])
        m_r01 = next(m for m in road_metrics if m.road_id == "R01")

        self.assertFalse(m_r01.is_hourly_rate_valid)
        self.assertIsNone(m_r01.expected_demand_vph)
        self.assertIsNone(m_r01.utilization_ratio)
        self.assertEqual(m_r01.status, RoadStatus.UNCALIBRATED)
        self.assertAlmostEqual(m_r01.expected_demand_in_window, 1.0, places=4)
        self.assertTrue(any(i.issue_type == "zero_duration_window" for i in issues))

    def test_audit_09_invalid_capacity_handling(self) -> None:
        """Audit 9: Zero, negative, and missing road capacity handled without ZeroDivisionError."""
        edge_r01 = self.graph.edges["R01"]
        edge_r02 = self.graph.edges["R02"]
        orig_cap_r01 = edge_r01.capacity_vph
        orig_cap_r02 = edge_r02.capacity_vph

        try:
            edge_r01.capacity_vph = 0.0
            edge_r02.capacity_vph = -100.0

            traj = NormalizedTrajectory(
                track_id="AUDIT_CAP",
                origin_node="J01",
                destination_node="J03",
                candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=1.0, metadata={"edges": ["R01", "R02"]})],
                vehicle_weight=1.0,
                time_window_start=1000.0,
                time_window_end=1360.0,
            )
            road_metrics, _, _, _ = self.flow_engine.aggregate_flows([traj])
            m_r01 = next(m for m in road_metrics if m.road_id == "R01")
            m_r02 = next(m for m in road_metrics if m.road_id == "R02")

            self.assertEqual(m_r01.capacity_vph, 0.0)
            self.assertEqual(m_r01.utilization_ratio, 0.0)
            self.assertEqual(m_r02.capacity_vph, 0.0)
            self.assertEqual(m_r02.utilization_ratio, 0.0)
        finally:
            edge_r01.capacity_vph = orig_cap_r01
            edge_r02.capacity_vph = orig_cap_r02

    def test_audit_10_real_data_no_fabrication(self) -> None:
        """Audit 10: Verify real data pipeline refuses to fabricate cross-camera trajectories."""
        from inference import load_observations_from_json, IdentityGraph
        real_file = Path("data/observations/kanishka_traffic.json")
        camera_metadata = {"traffic": {"latitude": 17.3850, "longitude": 78.4867}}
        observations = load_observations_from_json(real_file, camera_metadata=camera_metadata)

        self.assertEqual(len(observations), 2503)

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph(observations, camera_metadata=camera_metadata)
        candidate_identities = graph.get_candidate_identities()

        self.assertEqual(len(candidate_identities), 2503)
        multi_cams = [c for c in candidate_identities if len(c.get("observation_ids", [])) > 1]
        self.assertEqual(len(multi_cams), 0)


if __name__ == "__main__":
    unittest.main()
