"""
UrbanTrack AI — AI City 2022 Track 1 Comprehensive Ground-Truth Validation Runner.

Executes end-to-end scientific validation:
1. Ingestion & Normalization of Real 3-Camera Perception Data
2. Official Synchronization Attachment (AICitySynchronizer)
3. Official Homography Calibration & World-Coordinate Transformation (AICityCalibration)
4. Leakage-Free Inference (Candidate Generation, Multimodal Fusion, Identity Graph, Trajectories)
5. Automated Zero-Leakage Audit
6. Spatiotemporal Ground-Truth Linking & Mapping (AICityGroundTruthAdapter)
7. Supervised Multi-Camera Identity & Candidate Recall Evaluation
8. World-Space Trajectory Plausibility & Transit Time Evaluation
9. Probabilistic Calibration with Disjoint Vehicle-Level Holdout Split (Brier, ECE)
10. 5-Way Ablation Benchmark
11. Controlled Robustness Perturbation Benchmarks
12. Scalability Profiling
13. Complete Artifact Serialization to results/aicity_validation/
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.observation_loader import load_aicity_member1_feed
from inference.road_graph import RoadGraph
from inference.trajectory_engine import reconstruct_identity_trajectory
from schemas.observation_schema import Observation

from inference.aicity_synchronizer import AICitySynchronizer
from inference.aicity_calibration import AICityCalibration
from inference.aicity_gt_adapter import AICityGroundTruthAdapter
from inference.aicity_evaluator import (
    audit_inference_leakage,
    evaluate_candidate_recall,
    evaluate_identity_associations,
    evaluate_world_space_trajectories,
    evaluate_probabilistic_calibration,
    run_ablation_experiments,
    run_robustness_experiments,
    run_scalability_benchmark,
)


def run_aicity_validation(
    results_dir: str | Path = "results/aicity_validation",
    handoff_dir: str | Path = "UrbanTrack_Member1_Handoff/output",
    sync_file: str | Path = "data/aicity_ground_truth/cam_timestamp/S01.txt",
    cal_dir: str | Path = "data/aicity_ground_truth/calibration",
    gt_dir: str | Path = "data/aicity_ground_truth/gt",
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================================")
    print("      URBANTRACK AI — AI CITY 2022 GROUND-TRUTH VALIDATION & WORLD TRAJECTORY PIPELINE    ")
    print("=========================================================================================\n")

    # 1. Ingestion
    print("--- 1. MULTI-CAMERA DATASET INGESTION ---")
    observations = load_aicity_member1_feed(output_dir=handoff_dir, fps=10.0)
    discovered_cams = sorted(list({o.camera_id for o in observations}))
    print(f"  - Ingestion Directory            : {handoff_dir}")
    print(f"  - Discovered Cameras             : {discovered_cams}")
    print(f"  - Total Ingested Tracklets       : {len(observations)}")

    # 2. Synchronization
    print("\n--- 2. OFFICIAL SYNCHRONIZATION ATTACHMENT ---")
    synchronizer = AICitySynchronizer(sync_file=sync_file)
    observations = synchronizer.attach_synchronization(observations)
    print(f"  - Synchronization Source         : {sync_file}")
    for cam, offset in synchronizer.offsets.items():
        if cam.startswith("CAM_"):
            print(f"    * {cam}: start offset = {offset:+.3f} s")

    # 3. Calibration & World Coordinates
    print("\n--- 3. OFFICIAL CALIBRATION & WORLD COORDINATE PROJECTION ---")
    calibrator = AICityCalibration(calibration_dir=cal_dir)
    observations = calibrator.attach_calibration(observations)
    calibrated_obs_count = sum(1 for o in observations if getattr(o, "world_x", None) is not None)
    print(f"  - Calibration Source Directory   : {cal_dir}")
    print(f"  - Calibrated Cameras             : {calibrator.to_dict()['cameras_calibrated']}")
    print(f"  - Observations with World Coords : {calibrated_obs_count} / {len(observations)} ({calibrated_obs_count/len(observations)*100:.1f}%)")
    print(f"  - Spatial Coordinate System Tag  : 'cityflow_world' (meters; zero fabricated GPS)")

    # 4. Leakage Audit BEFORE Inference
    print("\n--- 4. GROUND-TRUTH LEAKAGE AUDIT (PRE-INFERENCE) ---")
    gt_adapter = AICityGroundTruthAdapter(gt_dir=gt_dir)
    pre_leakage = audit_inference_leakage(observations, gt_adapter)
    print(f"  - Pre-Inference Leakage Status   : {pre_leakage['status'].upper()} (Zero GT attributes on observations)")
    assert pre_leakage["status"] == "passed", "Fatal: Ground-truth leakage detected on input observations!"

    # 5. Candidate Generation
    print("\n--- 5. CANDIDATE GENERATION & PRUNING ---")
    generator = CandidateGenerator(
        max_time_window_seconds=7200.0,
        min_probability_threshold=0.65,
        
    )
    t_cg_start = time.perf_counter()
    candidates, rejections = generator.generate_candidates(observations)
    t_cg_ms = (time.perf_counter() - t_cg_start) * 1000.0

    n_obs = len(observations)
    theoretical_pairs = (n_obs * (n_obs - 1)) // 2
    reduction_pct = ((theoretical_pairs - len(candidates)) / theoretical_pairs * 100.0) if theoretical_pairs > 0 else 0.0
    print(f"  - Theoretical Total Pairs        : {theoretical_pairs:,}")
    print(f"  - Candidate Pairs Generated      : {len(candidates):,} ({reduction_pct:.2f}% pruned in {t_cg_ms:.2f} ms)")
    print(f"  - Pruning Breakdown              : {rejections}")

    # 6. Multimodal Identity Fusion
    print("\n--- 6. MULTIMODAL IDENTITY FUSION (WITH C002 GUARD) ---")
    match_results = []
    c002_incompat_count = 0
    c001_c003_compat_count = 0

    for oa, ob in candidates:
        res = match_observations(oa, ob)
        res["obs_a_id"] = oa.observation_id
        res["obs_b_id"] = ob.observation_id
        match_results.append(res)

        app_status = res.get("evidence", {}).get("appearance_status")
        if app_status == "incompatible_models":
            c002_incompat_count += 1
        elif app_status == "available" and oa.camera_id != ob.camera_id:
            c001_c003_compat_count += 1

    decisions = Counter(r.get("decision_state") for r in match_results)
    print(f"  - Evaluated Candidate Pairs      : {len(match_results):,}")
    print(f"  - C002 Incompatible Re-ID Pairs  : {c002_incompat_count} (Safely guarded; cross-model similarity suppressed)")
    print(f"  - C001 ↔ C003 Compatible Pairs   : {c001_c003_compat_count}")
    print(f"  - Pairwise Decision States       : CONFIRMED={decisions['CONFIRMED']}, AMBIGUOUS={decisions['AMBIGUOUS']}, REJECTED={decisions['REJECTED']}")

    # 7. Identity Graph Clustering
    print("\n--- 7. IDENTITY GRAPH CLUSTERING ---")
    graph = IdentityGraph(min_score_threshold=0.70)
    graph.build_graph_from_matches(observations, candidates, match_results)
    clusters = graph.get_candidate_identities()
    cluster_statuses = Counter(c.get("identity_status") for c in clusters)
    print(f"  - Identity Graph Nodes / Edges   : {len(graph.nodes)} nodes / {len(graph.edges)} edges")
    print(f"  - Inferred Vehicle Clusters      : {len(clusters)} total {dict(cluster_statuses)}")

    # 8. Trajectory Inference (World-Space & Image-Space)
    print("\n--- 8. WORLD-SPACE TRAJECTORY INFERENCE ---")
    dummy_road_graph = RoadGraph()
    trajectories_output: List[Dict[str, Any]] = []

    for idx, cl in enumerate(clusters):
        ut_id = f"UT_ID_{idx + 1:04d}"
        cl["candidate_vehicle_id"] = ut_id
        cl["identity_id"] = ut_id

        traj = reconstruct_identity_trajectory(
            identity_data=cl,
            road_graph=dummy_road_graph,
            config={"mode": "world_space", "fallback_enabled": True},
        )
        traj_dict = traj.to_dict()
        traj_dict["spatial_semantics"] = "cityflow_world"
        traj_dict["temporal_semantics"] = "aicity_official_synchronized"
        trajectories_output.append(traj_dict)

    print(f"  - Reconstructed Trajectories     : {len(trajectories_output)}")

    # 9. Ground-Truth Linking (Strict Post-Inference Step)
    print("\n--- 9. GROUND-TRUTH TRACKLET LINKING & VALIDATION ---")
    assocs = gt_adapter.link_member1_tracklets(observations, handoff_output_dir=handoff_dir)
    gt_summary = gt_adapter.to_dict()
    print(f"  - Official GT Vehicles in Scene  : {gt_summary['total_official_gt_vehicles']}")
    print(f"  - Member 1 Tracklets Mapped      : {gt_summary['mapped_observations_count']} / {len(observations)} ({gt_summary['mapped_observations_count']/len(observations)*100:.1f}%)")
    print(f"  - Multi-Camera Vehicles in GT    : {gt_summary['multi_camera_gt_vehicles_count']}")

    # 10. Candidate Recall & Identity Evaluation
    print("\n--- 10. SUPERVISED EVALUATION METRICS ---")
    cand_recall_metrics = evaluate_candidate_recall(candidates, gt_adapter, observations)
    identity_metrics_cross = evaluate_identity_associations(
        predicted_pairs=match_results,
        clusters=clusters,
        gt_adapter=gt_adapter,
        decision_threshold=0.70,
        cross_camera_only=True,
    )
    identity_metrics_all = evaluate_identity_associations(
        predicted_pairs=match_results,
        clusters=clusters,
        gt_adapter=gt_adapter,
        decision_threshold=0.70,
        cross_camera_only=False,
    )

    print(f"  - Candidate Recall on Cross-Cam  : {cand_recall_metrics.get('candidate_recall_pct')}% ({cand_recall_metrics.get('retrieved_true_cross_pairs')}/{cand_recall_metrics.get('total_true_cross_pairs')} true GT pairs retained)")
    print(f"  - Cross-Camera Precision         : {identity_metrics_cross['precision']:.4f}")
    print(f"  - Cross-Camera Recall            : {identity_metrics_cross['recall']:.4f}")
    print(f"  - Cross-Camera F1 Score          : {identity_metrics_cross['f1_score']:.4f}")
    print(f"  - Cross-Camera False Merge Rate  : {identity_metrics_cross['false_merge_rate']:.6f} (Strict safety against false positive merges)")
    print(f"  - Overall Cluster Purity         : {identity_metrics_all['cluster_purity']:.4f}")

    # 11. World-Space Trajectory Evaluation
    print("\n--- 11. WORLD-SPACE TRAJECTORY EVALUATION ---")
    traj_metrics = evaluate_world_space_trajectories(trajectories_output, observations, gt_adapter)
    print(f"  - Transitions with World Coords  : {traj_metrics['transitions_with_world_coordinates']}")
    print(f"  - Total World Distance Traversed : {traj_metrics['total_world_distance_meters']:,.1f} meters")
    print(f"  - Mean Transition Speed          : {traj_metrics['mean_transition_speed_kmh']:.1f} km/h (Median: {traj_metrics['median_transition_speed_kmh']:.1f} km/h)")
    print(f"  - Speed Feasibility Rate (<120)  : {traj_metrics['speed_plausibility_rate']*100:.1f}%")

    # 12. Probabilistic Holdout Calibration
    print("\n--- 12. PROBABILISTIC CALIBRATION (DISJOINT VEHICLE HOLDOUT) ---")
    calib_metrics = evaluate_probabilistic_calibration(match_results, gt_adapter, dev_ratio=0.60, seed=42)
    print(f"  - Calibration Status             : {calib_metrics['status'].upper()}")
    if calib_metrics["status"] == "passed":
        print(f"  - Vehicle-Level Split            : {calib_metrics['dev_vehicles_count']} DEV / {calib_metrics['holdout_vehicles_count']} HOLDOUT vehicles (Zero leakage)")
        print(f"  - Uncalibrated Brier / ECE       : Brier={calib_metrics['uncalibrated_brier_score']:.4f} | ECE={calib_metrics['uncalibrated_ece']:.4f}")
        print(f"  - Calibrated Brier / ECE         : Brier={calib_metrics['calibrated_brier_score']:.4f} | ECE={calib_metrics['calibrated_ece']:.4f}")
        print(f"  - Calibration Improvement        : Brier reduced by {calib_metrics['brier_score_improvement']:+.4f} | ECE reduced by {calib_metrics['ece_reduction']:+.4f}")

    # 13. Ablation Study
    print("\n--- 13. 5-WAY ABLATION BENCHMARK ---")
    ablation_results = run_ablation_experiments(observations, gt_adapter)
    for b_name, b_data in ablation_results.items():
        print(f"  * {b_name:<38}: P={b_data['precision']:.4f} | R={b_data['recall']:.4f} | F1={b_data['f1_score']:.4f} | FMR={b_data['false_merge_rate']:.6f} | Purity={b_data['cluster_purity']:.4f}")

    # 14. Robustness Experiments
    print("\n--- 14. CONTROLLED ROBUSTNESS BENCHMARKS ---")
    robustness_results = run_robustness_experiments(observations, gt_adapter, base_f1=identity_metrics_cross["f1_score"])
    for r_name, r_data in robustness_results.items():
        print(f"  * {r_name:<30}: F1={r_data['perturbed_f1']:.4f} (delta={r_data['delta_f1']:+.4f}) | Purity={r_data['cluster_purity']:.4f} | {r_data['failure_mode']}")

    # 15. Scalability Profiling
    print("\n--- 15. SCALABILITY PROFILING ---")
    scalability_results = run_scalability_benchmark(observations, scales=[50, 100, 200, 384, 500, 1000])
    for s_item in scalability_results:
        print(f"  * N={s_item['observation_count']:4d}: {s_item['theoretical_pairs']:7,d} theoretical -> {s_item['generated_candidate_pairs']:6,d} candidates ({s_item['search_space_reduction_pct']:5.2f}% pruned in {s_item['candidate_generation_runtime_ms']:6.2f} ms)")

    # 16. Serialization of All 14 Artifacts
    print("\n--- 16. SERIALIZING MACHINE-READABLE VALIDATION ARTIFACTS ---")

    tracklet_to_gt_data = {
        obs_id: assoc.to_dict() for obs_id, assoc in assocs.items()
    }

    inferred_identities_data = [
        {
            "identity_id": cl.get("candidate_vehicle_id"),
            "identity_status": cl.get("identity_status"),
            "admission_status": cl.get("admission_status"),
            "cameras": cl.get("cameras_visited", []),
            "observation_ids": cl.get("observation_ids", []),
            "observations_count": len(cl.get("observation_ids", [])),
            "vehicle_type": (cl.get("member_observations") or [{}])[0].get("vehicle_type"),
            "confidence": cl.get("identity_confidence"),
            "consistency": cl.get("cluster_consistency"),
            "evidence_summary": cl.get("identity_evidence_summary"),
        }
        for cl in clusters
    ]

    artifacts_map = {
        "dataset_inventory.json": {
            "total_observations": len(observations),
            "cameras": discovered_cams,
            "calibrated_observations": calibrated_obs_count,
            "synchronized_observations": len(observations),
            "timestamp_semantics": "video_relative",
            "synchronized_timestamp_semantics": "aicity_official_synchronized",
            "coordinate_system": "cityflow_world",
        },
        "synchronization_metadata.json": synchronizer.to_dict(),
        "calibration_metadata.json": calibrator.to_dict(),
        "ground_truth_inventory.json": gt_summary,
        "tracklet_to_gt.json": tracklet_to_gt_data,
        "inference_results.json": {
            "candidate_pairs_evaluated": len(candidates),
            "pairwise_decisions": dict(decisions),
            "inferred_clusters_count": len(clusters),
            "cluster_status_distribution": dict(cluster_statuses),
            "inferred_identities": inferred_identities_data,
        },
        "identity_metrics.json": {
            "candidate_recall": cand_recall_metrics,
            "cross_camera_identity_metrics": identity_metrics_cross,
            "overall_identity_metrics": identity_metrics_all,
        },
        "trajectory_metrics.json": traj_metrics,
        "calibration_metrics.json": calib_metrics,
        "ablation_results.json": ablation_results,
        "robustness_results.json": robustness_results,
        "scalability_results.json": scalability_results,
        "leakage_audit.json": pre_leakage,
        "provenance.json": {
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_handoff_dir": str(handoff_dir),
            "sync_file": str(sync_file),
            "calibration_dir": str(cal_dir),
            "gt_dir": str(gt_dir),
            "zero_gt_leakage_enforced": True,
            "zero_data_fabrication_enforced": True,
        },
    }

    for fname, data in artifacts_map.items():
        p_file = out_dir / fname
        with open(p_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  - Serialized                     : {p_file} ({p_file.stat().st_size:,} bytes)")

    t_total = time.perf_counter() - t_start
    print(f"\nAI City 2022 Validation completed successfully in {t_total:.2f} seconds.")

    return {
        "status": "success",
        "runtime_seconds": t_total,
        "artifacts_generated": list(artifacts_map.keys()),
        "identity_f1": identity_metrics_cross["f1_score"],
        "candidate_recall": cand_recall_metrics.get("candidate_recall"),
        "calibrated_observations": calibrated_obs_count,
        "total_gt_vehicles": gt_summary["total_official_gt_vehicles"],
    }


if __name__ == "__main__":
    run_aicity_validation()
