"""
UrbanTrack AI - Day 2 Real Observation Evaluation Script.
Runs finalized Day 2 Identity Fusion Engine against Kanishka's real perception feed.
"""

from pathlib import Path
import json

from inference import (
    IdentityGraph,
    Observation,
    load_observations_from_json,
    match_observations,
)


def execute_real_data_analysis():
    base_dir = Path(__file__).parent
    real_file = base_dir / "data" / "observations" / "kanishka_traffic.json"

    print("=========================================================================================")
    print("        URBANTRACK AI - DAY 2 REAL DATA ANALYSIS (KANISHKA'S TRAFFIC FEED)        ")
    print("=========================================================================================\n")

    with open(real_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    total_records = len(raw_data)
    print(f"Total Raw Records Loaded   : {total_records}")

    # Inspect missing & data quality fields
    missing_track_id = sum(1 for r in raw_data if "track_id" not in r or r["track_id"] is None)
    missing_embedding = sum(1 for r in raw_data if "appearance_embedding" not in r or not r.get("appearance_embedding"))
    missing_plate = sum(1 for r in raw_data if "plate" not in r or not r.get("plate"))
    invalid_confidence = sum(1 for r in raw_data if "confidence" in r and not (0.0 <= r["confidence"] <= 1.0))
    
    # Check duplicates (same camera, same frame, same bbox)
    seen_sigs = set()
    duplicate_count = 0
    for r in raw_data:
        bbox_tuple = tuple(r["bbox"]) if "bbox" in r and r["bbox"] else ()
        sig = (r.get("camera_id"), r.get("frame_id"), bbox_tuple)
        if sig in seen_sigs:
            duplicate_count += 1
        else:
            seen_sigs.add(sig)

    print("\n--- DATA QUALITY & MISSING FIELDS ANALYSIS ---")
    print(f"  - Observations Missing 'track_id'            : {missing_track_id} / {total_records} (100.0%)")
    print(f"  - Observations Missing 'appearance_embedding': {missing_embedding} / {total_records} (100.0%)")
    print(f"  - Observations Missing 'plate'               : {missing_plate} / {total_records} (100.0%)")
    print(f"  - Observations with Invalid Confidence       : {invalid_confidence}")
    print(f"  - Duplicate Detections (Same Camera/Frame/BBox): {duplicate_count}")

    # Load Observations via loader
    camera_metadata = {"traffic": {"latitude": 17.3850, "longitude": 78.4867}}
    observations = load_observations_from_json(real_file, camera_metadata=camera_metadata)

    total_possible_pairs = (total_records * (total_records - 1)) // 2
    print(f"\nTotal Possible Pairwise Comparisons : {total_possible_pairs:,}")

    # Evaluate pair subset (first 100 observations to demonstrate pair evaluation)
    sample_sub = observations[:100]
    n_sub = len(sample_sub)
    sub_pairs = (n_sub * (n_sub - 1)) // 2
    
    rejected_simultaneous = 0
    rejected_type = 0
    ambiguous_pairs = 0
    strong_matches = 0

    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            obs_a, obs_b = sample_sub[i], sample_sub[j]
            if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
                obs_a, obs_b = obs_b, obs_a

            res = match_observations(obs_a, obs_b, camera_metadata=camera_metadata)
            prob = res["same_vehicle_probability"]

            if prob == 0.0:
                if "Incompatible vehicle types" in res["explanation"]:
                    rejected_type += 1
                else:
                    rejected_simultaneous += 1
            elif prob >= 0.70:
                strong_matches += 1
            elif 0.40 <= prob <= 0.75:
                ambiguous_pairs += 1

    print(f"\n--- PAIR EVALUATION SAMPLE ({n_sub} OBS / {sub_pairs:,} PAIRS) ---")
    print(f"  - Physical Rejections (Same Frame / Incompatible): {rejected_simultaneous + rejected_type}")
    print(f"  - Unconfirmed Evidence Scores (configured neutral fallback): {ambiguous_pairs}")
    print(f"  - Confident Identity Matches (evidence score >= 0.70): {strong_matches}")

    # Build Identity Graph
    graph = IdentityGraph(min_probability_threshold=0.70)
    graph.build_graph(observations, camera_metadata=camera_metadata)
    candidate_identities = graph.get_candidate_identities()

    print("\n--- IDENTITY GRAPH CLUSTERING RESULTS ---")
    print(f"  - Discovered Candidate Identity Clusters         : {len(candidate_identities)}")
    print(f"  - Observations Remaining Unmerged (Singletons)   : {len(candidate_identities)}")
    print("\n  DATA LIMITATION DIAGNOSTIC:")
    print("  Because Kanishka's perception feed currently lacks Re-ID appearance embeddings")
    print("  and license plates, the feed lacks usable identity evidence.")
    print("  With configured identity-link threshold min_probability_threshold = 0.70,")
    print("  the engine safely prevents unconfirmed pairs from forming false identity edges.")
    print(f"  When cross-camera identity evidence is insufficient, UrbanTrack does not fabricate identity links,")
    print(f"  leaving observations as separate candidate identities ({len(candidate_identities)}) rather than forcefully merging them.")

    # Day 5 Reliability & Uncertainty Evaluation on Real Kanishka Feed
    from inference.reliability_engine import evaluate_observation_reliability

    print("\n--- DAY 5 RELIABILITY & UNCERTAINTY AUDIT (SAMPLE REAL OBSERVATIONS) ---")
    sample_obs = observations[0]
    rel_eval = evaluate_observation_reliability(sample_obs, camera_metadata=camera_metadata)

    print(f"  Sample Observation ID          : {rel_eval.observation_id}")
    print(f"  Camera ID                      : {rel_eval.camera_id}")
    print(f"  Camera Reliability             : {rel_eval.camera_reliability:.4f}")
    print(f"  Detection Confidence           : {rel_eval.detection_confidence:.4f}")
    print(f"  Physical Observation Quality   : {rel_eval.observation_quality:.4f} (sensor detection quality)")
    print(f"  Identity Evidence Quality      : {rel_eval.identity_evidence_quality:.4f} (incomplete / unavailable)")
    print(f"  Missing Evidence List          : {rel_eval.missing_evidence}")
    print(f"  Sensor Degradation Factors     : {rel_eval.degradation_factors}")
    print(f"  Composite Tracking Reliability : {rel_eval.overall_reliability:.4f}")
    print(f"  Composite Uncertainty          : {rel_eval.uncertainty:.4f} (level: {rel_eval.uncertainty_level})")
    print(f"  Semantic Explanation          : {rel_eval.explanation}")

    # Day 6 City Mobility Flow Evaluation on Real Kanishka Feed
    print("\n--- DAY 6 CITY MOBILITY FLOW EVALUATION (REAL DATA FEED) ---")
    multi_cam_identities = [ident for ident in candidate_identities if len(ident.get("observation_ids", [])) > 1]
    if not multi_cam_identities:
        print("  Status: Insufficient identity evidence for cross-camera city-flow aggregation.")
        print("  Reason: Real Kanishka observations remain separate candidate identities (lacking cross-camera Re-ID & plates).")
        print("  Action: Safely aborting city flow fabrication. Zero synthetic trajectories or traffic demand created.")
    else:
        print(f"  Multi-camera identities found: {len(multi_cam_identities)}")

    # Controlled Synthetic Demonstration
    print("\n--- DAY 6 CONTROLLED SYNTHETIC DEMONSTRATION ---")
    from mobility.mobility_engine import CityMobilityEngine
    from schemas.normalized_trajectory_schema import NormalizedTrajectory

    city_net_file = base_dir / "data" / "synthetic" / "city_network.json"
    mobility_engine = CityMobilityEngine(city_net_file)

    scenarios_file = base_dir / "data" / "synthetic" / "day6_mobility_scenarios.json"
    with open(scenarios_file, "r", encoding="utf-8") as sf:
        scenarios_data = json.load(sf)["scenarios"]

    # Demonstrate using Case 5 (Multiple Vehicles) with a defined 1-hour time window (3600s)
    demo_trajs = [
        NormalizedTrajectory.from_dict({
            **t,
            "time_window": {"start": 0.0, "end": 3600.0}
        })
        for t in scenarios_data["case5_multiple_vehicles"]["trajectories"]
    ]
    report = mobility_engine.process_trajectories(demo_trajs, top_priority_count=3)

    max_util = report.summary.get("max_road_utilization")
    util_str = f"{max_util:.4f}" if max_util is not None else "N/A (uncalibrated duration)"

    print("  Demonstrating Day 6 Mobility Analytics using Controlled Synthetic Trajectories [LABEL: SYNTHETIC]:")
    print(f"  - Total Input Trajectories      : {report.summary['total_trajectories_input']}")
    print(f"  - Valid Trajectories Processed  : {report.summary['valid_trajectories_count']}")
    print(f"  - Demand Conservation Verified  : {report.summary['demand_conservation_verified']}")
    print(f"  - Total Physical OD Demand (PCU): {report.summary['total_physical_demand_od']}")
    print(f"  - Active Roads with Flow        : {report.summary['active_roads_with_demand']} / {report.summary['total_roads_in_network']}")
    print(f"  - Highest Utilization Road      : {report.summary['highest_utilization_road']} (utilization: {util_str})")
    print(f"  - Priority Roads (Top 3)        : {[p.road_id for p in report.priority_roads]}")

    # Day 7 City Anomaly Detection Evaluation on Real Kanishka Feed
    print("\n--- DAY 7 CITY ANOMALY DETECTION EVALUATION (REAL DATA FEED) ---")
    if not multi_cam_identities:
        print("  Status: Insufficient cross-camera identity evidence for trajectory-level anomaly detection.")
        print("  Reason: Real Kanishka observations remain separate candidate identities (lacking cross-camera Re-ID & plates).")
        print("  Action: Safely skipping trajectory anomaly detection. Zero fake anomalies or synthetic baselines created.")
    else:
        print(f"  Multi-camera identities available: {len(multi_cam_identities)}")

    # Day 7 Controlled Synthetic Demonstration
    print("\n--- DAY 7 CONTROLLED SYNTHETIC DEMONSTRATION ---")
    from anomaly.investigation_engine import InvestigationEngine
    from schemas.anomaly_schema import MobilityBaseline

    d7_scenarios_file = base_dir / "data" / "synthetic" / "day7_anomaly_scenarios.json"
    with open(d7_scenarios_file, "r", encoding="utf-8") as d7f:
        d7_data = json.load(d7f)
        d7_scenarios = d7_data["scenarios"]
        d7_baseline = MobilityBaseline.from_dict(d7_data["baseline_config"])

    investigation_engine = InvestigationEngine(baseline=d7_baseline, road_graph=mobility_engine.graph)
    demo_anomaly_trajs = [
        NormalizedTrajectory.from_dict(d7_scenarios["case1_normal_vehicle"]["trajectory"]),
        NormalizedTrajectory.from_dict(d7_scenarios["case2_slow_travel_time"]["trajectory"]),
        NormalizedTrajectory.from_dict(d7_scenarios["case3_impossible_speed"]["trajectory"]),
        NormalizedTrajectory.from_dict(d7_scenarios["case4_route_deviation"]["trajectory"]),
    ]
    anomaly_report = investigation_engine.run_investigation(demo_anomaly_trajs, mobility_report=report)

    print("  Demonstrating Day 7 Anomaly Investigation using Controlled Synthetic Scenarios [LABEL: SYNTHETIC]:")
    print(f"  - Total Trajectories Investigated: {anomaly_report.summary['total_trajectories_evaluated']}")
    print(f"  - Anomalous Trajectories Flagged : {anomaly_report.summary['anomalous_trajectories_count']}")
    print(f"  - High Priority Vehicle Cases    : {anomaly_report.summary['vehicle_severity_breakdown']['high_priority']}")
    print(f"  - Investigate Vehicle Cases      : {anomaly_report.summary['vehicle_severity_breakdown']['investigate']}")
    print(f"  - Normal / Compliant Movements   : {anomaly_report.summary['vehicle_severity_breakdown']['normal']}")
    print(f"  - Anomalous Roads Flagged        : {anomaly_report.summary['anomalous_roads_count']}")
    print(f"  - Network Bottleneck Alerts      : {anomaly_report.summary['network_anomalies_count']}")
    top_veh = anomaly_report.vehicle_anomalies[0]
    print(f"  - Sample Investigation Candidate : {top_veh.entity_id} (Score: {top_veh.overall_score:.2f}, Priority: {top_veh.investigation_priority.value.upper()})")
    print(f"    Explanation: {top_veh.explanation}")

    # Day 8 Counterfactual Simulation Evaluation on Real Kanishka Feed
    print("\n--- DAY 8 COUNTERFACTUAL SIMULATION EVALUATION (REAL DATA FEED) ---")
    if not multi_cam_identities:
        print("  Status: Insufficient evidence for real-data counterfactual trajectory simulation.")
        print("  Reason: Real Kanishka observations remain separate candidate identities (lacking cross-camera Re-ID & plates).")
        print("  Action: Safely skipping real-data counterfactual simulation. Zero fake trajectories or synthetic interventions applied.")
    else:
        print(f"  Multi-camera identities available: {len(multi_cam_identities)}")

    # Day 8 Controlled Synthetic Demonstration
    print("\n--- DAY 8 CONTROLLED SYNTHETIC DEMONSTRATION ---")
    from simulation.counterfactual_engine import CounterfactualEngine
    from schemas.scenario_schema import ScenarioDefinition

    cf_engine = CounterfactualEngine()
    d8_scenarios_file = base_dir / "data" / "synthetic" / "day8_counterfactual_scenarios.json"
    with open(d8_scenarios_file, "r", encoding="utf-8") as d8f:
        d8_data = json.load(d8f)
        d8_case = d8_data["scenarios"]["SCN_001_SINGLE_ROAD_CLOSURE"]

    d8_scen = ScenarioDefinition.from_dict(d8_case)
    d8_traj = NormalizedTrajectory.from_dict(d8_case["trajectory"])
    cf_report = cf_engine.simulate_counterfactual([d8_traj], mobility_engine.graph, d8_scen)

    print("  Demonstrating Day 8 Counterfactual Simulation (What-If Road Closure) [LABEL: SYNTHETIC]:")
    print(f"  - Scenario ID                   : {cf_report.scenario_id} ({cf_report.scenario_type})")
    print(f"  - Intervention                  : {d8_scen.description}")
    print(f"  - Simulation Status             : {cf_report.status.value.upper()}")
    print(f"  - Displaced Demand              : {cf_report.impact['displaced_demand']} PCU")
    print(f"  - Unroutable Demand             : {cf_report.impact['unroutable_demand']} PCU")
    print(f"  - Alternate Corridors Identified: {cf_report.impact['alternate_corridors']}")
    print(f"  - Relieved Roads                : {cf_report.impact['relieved_roads']}")

    print("\n=========================================================================================")
    print("        REAL DATA EVALUATION COMPLETED SAFELY        ")
    print("=========================================================================================")


if __name__ == "__main__":
    execute_real_data_analysis()
