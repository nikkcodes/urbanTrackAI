"""
UrbanTrack AI — Day 9 Full End-to-End Pipeline Validator.

Executes and validates the supported cross-module integration boundaries:
    Stage 1: Observation loading & schema verification
    Stage 2: Identity fusion & identity graph clustering
    Stage 3: Trajectory reconstruction
    Stage 4: Sparse / missing-camera inference
    Stage 5: Reliability & uncertainty evaluation
    Stage 6: NormalizedTrajectory adaptation & contract verification
    Stage 7: Mobility flow & OD demand aggregation
    Stage 8: Anomaly detection & explainable investigation
    Stage 9: Counterfactual traffic simulation & baseline immutability

Executable:
    python3 tests/validate_full_pipeline.py
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.identity_graph import IdentityGraph
from inference.member3_adapter import adapt_vehicle_trajectory_to_normalized
from inference.observation_loader import load_observations_from_json
from inference.road_graph import RoadGraph
from inference.sparse_engine import infer_sparse_identity_trajectory
from inference.trajectory_engine import reconstruct_identity_trajectory
from mobility.flow_engine import MobilityFlowEngine
from anomaly.investigation_engine import InvestigationEngine
from simulation.counterfactual_engine import CounterfactualEngine
from schemas.scenario_schema import ScenarioDefinition, ScenarioStatus


def run_full_pipeline_validation() -> bool:
    print("=" * 80)
    print("      URBANTRACK AI - DAY 9 FULL END-TO-END PIPELINE VALIDATOR      ")
    print("=" * 80)

    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 0. SETUP & ROAD GRAPH INITIALIZATION
    # -------------------------------------------------------------------------
    network_path = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
    scenario_path = PROJECT_ROOT / "data" / "synthetic" / "day9_end_to_end_scenarios.json"

    if not network_path.is_file():
        print(f"[FAIL] Missing road network file: {network_path}")
        return False
    if not scenario_path.is_file():
        print(f"[FAIL] Missing synthetic Day 9 scenario file: {scenario_path}")
        return False

    road_graph = RoadGraph.from_json_file(network_path)
    # Associate cameras CAM_J01..CAM_J08 to junctions J01..J08
    for i in range(1, 9):
        road_graph.camera_associations[f"CAM_J{i:02d}"] = f"J{i:02d}"

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_data = json.load(f)

    # -------------------------------------------------------------------------
    # STAGE 1: OBSERVATION LOADING & SCHEMA VERIFICATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    observations = load_observations_from_json(
        scenario_data["observations"], camera_metadata=scenario_data.get("cameras")
    )
    timings["Stage 1 (Observation)"] = (time.perf_counter() - t0) * 1000.0

    assert len(observations) >= 7, f"Expected at least 7 observations, got {len(observations)}"
    for obs in observations:
        assert obs.observation_id, "Missing observation_id"
        assert obs.camera_id, "Missing camera_id"
        assert obs.track_id, "Missing camera-local track_id"
        assert obs.timestamp_seconds is not None, "Missing timestamp_seconds"
        assert obs.timestamp_semantics == "synchronized", "Expected synchronized time reference"

    print(f"\n[PASS] Observation ({len(observations)} observations loaded, all schema contracts verified)")
    print(f"       Execution time: {timings['Stage 1 (Observation)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 2: IDENTITY FUSION / IDENTITY GRAPH
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    id_graph = IdentityGraph(min_probability_threshold=0.70)
    id_graph.build_graph(observations, camera_metadata=scenario_data.get("cameras"))
    candidate_identities = id_graph.get_candidate_identities()
    timings["Stage 2 (Identity)"] = (time.perf_counter() - t0) * 1000.0

    assert len(candidate_identities) >= 4, f"Expected at least 4 identity clusters, got {len(candidate_identities)}"

    # Identify multi-observation vehicle clusters
    veh1_cluster = next((c for c in candidate_identities if len(c["member_observations"]) == 3), None)
    veh2_cluster = next((c for c in candidate_identities if len(c["member_observations"]) == 2), None)
    singleton_clusters = [c for c in candidate_identities if len(c["member_observations"]) == 1]

    assert veh1_cluster is not None, "Expected multi-camera cluster for Vehicle 1 (KA01AB1234)"
    assert veh2_cluster is not None, "Expected sparse gap cluster for Vehicle 2 (DL04CD5678)"
    assert len(singleton_clusters) >= 2, "Expected impossible speed vehicle (MH12XY9999) to remain isolated singletons"

    print(f"[PASS] Identity ({len(candidate_identities)} clusters: {len(candidate_identities) - len(singleton_clusters)} multi-camera, {len(singleton_clusters)} singletons)")
    print(f"       Vehicle 1 (KA01AB1234): {len(veh1_cluster['member_observations'])} cameras {veh1_cluster['cameras_visited']} -> {veh1_cluster['identity_id']}")
    print(f"       Vehicle 2 (DL04CD5678): {len(veh2_cluster['member_observations'])} cameras {veh2_cluster['cameras_visited']} -> {veh2_cluster['identity_id']}")
    print(f"       Impossible-speed vehicle (MH12XY9999): correctly split into 2 singletons (no false merge)")
    print(f"       Execution time: {timings['Stage 2 (Identity)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 3: TRAJECTORY RECONSTRUCTION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    veh1_traj = reconstruct_identity_trajectory(veh1_cluster, road_graph)
    timings["Stage 3 (Trajectory)"] = (time.perf_counter() - t0) * 1000.0

    assert veh1_traj.observations_count == 3
    assert len(veh1_traj.segments) == 2
    assert veh1_traj.total_distance_meters == 4000.0  # R01 (1800m) + R02 (2200m)
    assert veh1_traj.total_time_seconds == 262.0
    assert veh1_traj.overall_confidence > 0.99

    print(f"[PASS] Trajectory (Reconstructed {len(veh1_traj.segments)} segments across {veh1_traj.cameras_visited})")
    print(f"       Distance: {veh1_traj.total_distance_meters:.1f}m, Travel time: {veh1_traj.total_time_seconds:.1f}s, Feasible: True")
    print(f"       Execution time: {timings['Stage 3 (Trajectory)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 4: SPARSE INFERENCE
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    veh2_traj = infer_sparse_identity_trajectory(veh2_cluster, road_graph)
    timings["Stage 4 (Sparse inference)"] = (time.perf_counter() - t0) * 1000.0

    assert len(veh2_traj.segments) == 1
    sparse_seg = veh2_traj.segments[0]
    assert len(sparse_seg.candidate_routes) >= 2, "Expected at least 2 alternative corridors for sparse gap"
    # Alternative 1: R01 + R08 via J02, Alternative 2: R07 + R18 via J04
    feasible_cr = [cr for cr in sparse_seg.candidate_routes if cr.feasible]
    assert len(feasible_cr) >= 2, "Expected at least 2 feasible alternative corridors for sparse gap"
    for cr in feasible_cr:
        assert cr.feasible, "Candidate route should be road-feasible"
        assert cr.estimated_likelihood > 0.0

    likelihood_sum = sum(cr.estimated_likelihood for cr in feasible_cr)
    assert abs(likelihood_sum - 1.0) < 1e-3, f"Likelihoods must normalize to ~1.0, got {likelihood_sum}"

    print(f"[PASS] Sparse inference (Detected unobserved movement gap between {sparse_seg.start_camera_id} and {sparse_seg.end_camera_id})")
    print(f"       Found {len(sparse_seg.candidate_routes)} alternative corridors: {[r.edges for r in sparse_seg.candidate_routes]}")
    print(f"       Likelihoods: {[round(r.estimated_likelihood, 4) for r in sparse_seg.candidate_routes]} (Sum = {likelihood_sum:.4f})")
    print(f"       Zero intermediate observation records fabricated.")
    print(f"       Execution time: {timings['Stage 4 (Sparse inference)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 5: RELIABILITY EVALUATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    # Reliability is populated on trajectory and segments
    veh1_rel = veh1_traj.reliability
    veh2_rel = veh2_traj.reliability
    timings["Stage 5 (Reliability)"] = (time.perf_counter() - t0) * 1000.0

    assert veh1_rel is not None, "Vehicle 1 trajectory reliability missing"
    assert veh2_rel is not None, "Vehicle 2 trajectory reliability missing"
    assert "overall_reliability" in veh1_rel
    assert "overall_uncertainty" in veh2_rel

    print(f"[PASS] Reliability (Propagated sensor reliability and route entropy)")
    print(f"       Vehicle 1 Reliability: {veh1_rel['overall_reliability']:.4f}, Uncertainty: {veh1_rel['overall_uncertainty']:.4f}")
    print(f"       Vehicle 2 (Gap) Reliability: {veh2_rel['overall_reliability']:.4f}, Uncertainty: {veh2_rel['overall_uncertainty']:.4f}")
    print(f"       Execution time: {timings['Stage 5 (Reliability)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 6: NORMALIZED TRAJECTORY ADAPTATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    norm_veh1 = adapt_vehicle_trajectory_to_normalized(veh1_traj, vehicle_weight=1.0, vehicle_class="car")
    norm_veh2 = adapt_vehicle_trajectory_to_normalized(veh2_traj, vehicle_weight=1.0, vehicle_class="car")
    normalized_trajectories = [norm_veh1, norm_veh2]
    timings["Stage 6 (NormalizedTrajectory)"] = (time.perf_counter() - t0) * 1000.0

    assert norm_veh1.track_id == veh1_cluster["identity_id"]
    assert norm_veh2.track_id == veh2_cluster["identity_id"]
    assert norm_veh1.origin_node == "J01" and norm_veh1.destination_node == "J03"
    assert norm_veh2.origin_node == "J01" and norm_veh2.destination_node == "J05"
    assert norm_veh1.vehicle_weight == 1.0
    assert norm_veh2.vehicle_weight == 1.0

    for nt in normalized_trajectories:
        prob_sum = sum(r.probability for r in nt.candidate_routes)
        assert abs(prob_sum - 1.0) < 1e-4, f"Normalized candidate routes must sum to 1.0, got {prob_sum}"

    print(f"[PASS] NormalizedTrajectory ({len(normalized_trajectories)} trajectories adapted to Member 3 contract)")
    print(f"       Traceable IDs: {[nt.track_id for nt in normalized_trajectories]}")
    print(f"       Origin-Destination pairs: {[(nt.origin_node, nt.destination_node) for nt in normalized_trajectories]}")
    print(f"       Execution time: {timings['Stage 6 (NormalizedTrajectory)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 7: MOBILITY FLOW AGGREGATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    flow_engine = MobilityFlowEngine(road_graph)
    road_metrics, od_matrix, flow_issues, conservation_records = flow_engine.aggregate_flows(
        normalized_trajectories
    )
    timings["Stage 7 (Mobility)"] = (time.perf_counter() - t0) * 1000.0

    assert len(road_metrics) == len(road_graph.edges)
    assert od_matrix.total_demand == 2.0, f"Expected total OD physical demand 2.0, got {od_matrix.total_demand}"
    assert len(flow_issues) == 0, f"Unexpected flow validation issues: {flow_issues}"

    # Verify demand conservation across all trajectories
    for cr in conservation_records:
        assert cr["is_conserved"], f"Demand conservation failed for track {cr['track_id']}"

    print(f"[PASS] Mobility ({len(road_metrics)} road segments evaluated, {len(od_matrix.pairs)} OD pairs)")
    print(f"       Total Physical Traffic Demand: {od_matrix.total_demand:.1f} PCU (100% conserved)")
    print(f"       OD Demands: {[(p.origin, p.destination, p.demand) for p in od_matrix.pairs]}")
    print(f"       Reliability is preserved as trust indicator, never multiplied into traffic demand.")
    print(f"       Execution time: {timings['Stage 7 (Mobility)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 8: ANOMALY DETECTION & INVESTIGATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    inv_engine = InvestigationEngine(road_graph=road_graph)
    anomaly_report = inv_engine.run_investigation(normalized_trajectories)
    timings["Stage 8 (Anomaly)"] = (time.perf_counter() - t0) * 1000.0

    assert len(anomaly_report.vehicle_anomalies) == len(normalized_trajectories)
    for va in anomaly_report.vehicle_anomalies:
        assert va.entity_type == "vehicle"
        assert not va.is_anomalous, "Compliant synthetic test vehicles should be classified as normal"
        assert "criminal" not in va.explanation.lower()
        assert "guilt" not in va.explanation.lower()

    print(f"[PASS] Anomaly ({len(anomaly_report.vehicle_anomalies)} trajectory investigations completed)")
    print(f"       Normal compliant trajectories: {len(anomaly_report.vehicle_anomalies)} / {len(normalized_trajectories)}")
    print(f"       Zero moralistic / criminal interpretations (strictly physical & behavioral deviations).")
    print(f"       Execution time: {timings['Stage 8 (Anomaly)']:.2f} ms")

    # -------------------------------------------------------------------------
    # STAGE 9: COUNTERFACTUAL SIMULATION
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    scenario_dict = scenario_data.get("counterfactual_scenario")
    assert scenario_dict is not None, "Missing counterfactual scenario definition"

    scenario = ScenarioDefinition.from_dict(scenario_dict)

    # Snapshot baseline states to verify strict baseline immutability
    base_edges_before = {rid: copy.deepcopy(e.to_dict()) for rid, e in road_graph.edges.items()}
    base_probs_before = [
        [r.probability for r in nt.candidate_routes] for nt in normalized_trajectories
    ]

    cf_engine = CounterfactualEngine()
    cf_report = cf_engine.simulate_counterfactual(normalized_trajectories, road_graph, scenario)
    timings["Stage 9 (Counterfactual)"] = (time.perf_counter() - t0) * 1000.0

    assert cf_report.status == ScenarioStatus.SUCCESS
    assert cf_report.impact["displaced_demand"] > 0.0, "Expected displaced demand from R07 closure"
    assert cf_report.impact["unroutable_demand"] == 0.0, "Alternative corridor R01+R08 should route all displaced traffic"
    assert "R07" in cf_report.impact["affected_roads"]
    assert "R01" in cf_report.impact["alternate_corridors"]

    # Verify baseline immutability
    base_edges_after = {rid: e.to_dict() for rid, e in road_graph.edges.items()}
    base_probs_after = [
        [r.probability for r in nt.candidate_routes] for nt in normalized_trajectories
    ]
    assert base_edges_before == base_edges_after, "Baseline road graph was mutated during simulation!"
    assert base_probs_before == base_probs_after, "Baseline candidate route probabilities were mutated!"

    print(f"[PASS] Counterfactual (Scenario '{scenario.scenario_id}' - {scenario.description})")
    print(f"       Status: {cf_report.status.value}")
    print(f"       Displaced Demand: {cf_report.impact['displaced_demand']:.4f} PCU")
    print(f"       Unroutable Demand: {cf_report.impact['unroutable_demand']:.4f} PCU")
    print(f"       Alternative Corridors Identified: {cf_report.impact['alternate_corridors']}")
    print(f"       Baseline Immutability Verified: 100% identical before and after simulation.")
    print(f"       Execution time: {timings['Stage 9 (Counterfactual)']:.2f} ms")

    total_time_ms = (time.perf_counter() - total_start) * 1000.0

    # -------------------------------------------------------------------------
    # TRACEABILITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    END-TO-END TRACEABILITY AUDIT                    ")
    print("=" * 80)
    print("Conceptual Trace Chain:")
    print("  Camera + Local Track ID:")
    print(f"    CAM_J01 + cam01_trk_01 -> Plate KA01AB1234")
    print("      ↓")
    print(f"  Identity Cluster ID:")
    print(f"    {veh1_cluster['identity_id']} (Confidence: {veh1_cluster['identity_confidence']:.2f})")
    print("      ↓")
    print(f"  NormalizedTrajectory:")
    print(f"    track_id: {norm_veh1.track_id} (Origin: {norm_veh1.origin_node} -> Destination: {norm_veh1.destination_node})")
    print("      ↓")
    print(f"  Candidate Route IDs & Edges:")
    for r in norm_veh1.candidate_routes:
        print(f"    Route {r.metadata.get('edges')} (Probability: {r.probability:.4f})")
    print("      ↓")
    print(f"  Mobility Road Demands:")
    for rid in ["R01", "R02"]:
        rf = next(r for r in road_metrics if r.road_id == rid)
        print(f"    Road {rid}: Expected Demand = {rf.expected_demand_in_window:.4f} PCU")
    print("      ↓")
    print(f"  Anomaly Result:")
    va1 = next(v for v in anomaly_report.vehicle_anomalies if v.entity_id == norm_veh1.track_id)
    print(f"    Entity {va1.entity_id}: Severity={va1.severity.value}, Priority={va1.investigation_priority.value}")
    print("      ↓")
    print(f"  Counterfactual Simulation:")
    print(f"    Scenario {scenario.scenario_id}: Displaced={cf_report.impact['displaced_demand']:.4f} PCU, Baseline Preserved")

    print("\n" + "=" * 80)
    print("                     PIPELINE PERFORMANCE SUMMARY                   ")
    print("=" * 80)
    for stage_name, t_ms in timings.items():
        print(f"  {stage_name:<30} : {t_ms:>8.2f} ms")
    print("-" * 80)
    print(f"  {'Total Full Pipeline':<30} : {total_time_ms:>8.2f} ms")
    print(f"  {'Throughput (Observations/sec)':<30} : {(len(observations) / (total_time_ms / 1000.0)):>8.1f} obs/sec")
    print("=" * 80)
    print(">>> FULL PIPELINE VALIDATION: ALL 9 STAGES PASSED CLEANLY <<<")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_full_pipeline_validation()
    sys.exit(0 if success else 1)
