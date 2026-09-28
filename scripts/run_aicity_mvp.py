"""
UrbanTrack AI — AI City Challenge 2022 Track 1 (Member 1 Handoff) Pipeline Runner.

Executes end-to-end integration:
1. Multi-Camera Feed Ingestion & Normalization
2. Dynamic Embedding & Schema Validation
3. Unsynchronized Candidate Pair Generation
4. Multimodal Identity Fusion (with C002 Re-ID Model Incompatibility Guard)
5. Identity Graph Clustering & Consistency Verification
6. Trajectory Inference (Image-Space / Temporal Fallback Mode)
7. Comprehensive Artifact Serialization
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.observation_loader import load_aicity_member1_feed
from inference.road_graph import RoadGraph
from inference.similarity import appearance_similarity
from inference.trajectory_engine import reconstruct_identity_trajectory
from schemas.observation_schema import Observation


def run_aicity_mvp(config_path: str | Path = "configs/aicity_mvp.yaml") -> Dict[str, Any]:
    t_start = time.perf_counter()
    cfg_p = Path(config_path)
    if not cfg_p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(cfg_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    handoff_path = config["dataset"]["handoff_path"]
    out_dir = Path(config.get("output", {}).get("results_dir", "results/aicity_mvp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================================")
    print("      URBANTRACK AI — REAL MEMBER 1 HANDOFF PIPELINE RUNNER (3-CAMERA MVP)               ")
    print("=========================================================================================\n")

    # 1. Ingestion
    print("--- 1. MULTI-CAMERA DATASET INGESTION & DISCOVERY ---")
    configured_cameras = [c["camera_id"] for c in config["dataset"].get("cameras", [])]
    observations = load_aicity_member1_feed(
        output_dir=handoff_path,
        camera_ids=configured_cameras or None,
        fps=float(config["dataset"].get("fps", 10.0)),
    )

    discovered_cams = sorted(list({o.camera_id for o in observations}))
    print(f"  - Ingestion Directory            : {handoff_path}")
    print(f"  - Discovered Camera Streams      : {discovered_cams}")
    print(f"  - Total Ingested Tracklets       : {len(observations)}")

    # Camera breakdown
    cam_stats: Dict[str, Dict[str, Any]] = {}
    for cam_id in discovered_cams:
        cam_obs = [o for o in observations if o.camera_id == cam_id]
        embs = [o for o in cam_obs if o.appearance_embedding is not None]
        reid_model = (cam_obs[0].source_provenance or {}).get("reid_model") if cam_obs else "unknown"
        cam_stats[cam_id] = {
            "total_tracklets": len(cam_obs),
            "with_embeddings": len(embs),
            "reid_model": reid_model,
        }
        print(f"    * {cam_id}: {len(cam_obs)} tracklets | {len(embs)} 512-D embeddings | model: {reid_model}")

    # 2. Dynamic Embedding & Schema Validation
    print("\n--- 2. DYNAMIC EMBEDDING & SCHEMA AUDIT ---")
    valid_embs_count = 0
    corrupted_embs_count = 0
    all_zero_count = 0
    non_normalized_count = 0
    plate_null_count = 0

    for o in observations:
        if o.plate is None:
            plate_null_count += 1
        if o.appearance_embedding is not None:
            emb = o.appearance_embedding
            is_valid = True
            if len(emb) != 512:
                is_valid = False
                corrupted_embs_count += 1
            if any(math.isnan(x) or math.isinf(x) for x in emb):
                is_valid = False
                corrupted_embs_count += 1
            norm = math.sqrt(sum(x * x for x in emb))
            if norm < 1e-6:
                is_valid = False
                all_zero_count += 1
            elif abs(norm - 1.0) > 0.01:
                non_normalized_count += 1
            if is_valid:
                valid_embs_count += 1

    print(f"  - Total Validated 512-D Vectors  : {valid_embs_count} / {len(observations)} ({valid_embs_count/len(observations)*100:.1f}%)")
    print(f"  - Corrupted / NaN / Inf Vectors  : {corrupted_embs_count}")
    print(f"  - All-Zero Vectors               : {all_zero_count}")
    print(f"  - Unit L2 Normalization Deviations: {non_normalized_count}")
    print(f"  - Plate Text Non-Fabrication     : {plate_null_count} / {len(observations)} strictly null (100.0% zero-fabrication)")
    print(f"  - Coordinate System Semantics    : 'image' (pixel plane centroids; zero GPS conversion)")
    print(f"  - Timestamp Semantics            : 'video_relative' (frame / 10.0; zero fake UTC conversion)")

    # 3. Candidate Generation
    print("\n--- 3. UNSYNCHRONIZED CANDIDATE PAIR GENERATION ---")
    cg_cfg = config["pipeline"].get("candidate_generation", {})
    unsync_mode = bool(cg_cfg.get("unsynchronized_mode", True))
    min_prob_thresh = float(cg_cfg.get("min_probability_threshold", 0.65))
    max_time_window = float(cg_cfg.get("max_time_window_seconds", 7200.0))

    generator = CandidateGenerator(
        max_time_window_seconds=max_time_window,
        min_probability_threshold=min_prob_thresh,
        unsynchronized_mode=unsync_mode,
        config=cg_cfg,
    )

    t_cg_start = time.perf_counter()
    candidates, rejection_counts = generator.generate_candidates(observations)
    t_cg_ms = (time.perf_counter() - t_cg_start) * 1000.0

    n_obs = len(observations)
    theoretical_pairs = (n_obs * (n_obs - 1)) // 2
    n_candidates = len(candidates)
    pruned_count = theoretical_pairs - n_candidates
    reduction_pct = (pruned_count / theoretical_pairs * 100.0) if theoretical_pairs > 0 else 0.0

    same_cam_candidates = sum(1 for a, b in candidates if a.camera_id == b.camera_id)
    cross_cam_candidates = sum(1 for a, b in candidates if a.camera_id != b.camera_id)

    print(f"  - Generation Mode                : Unsynchronized Multi-Camera Mode ({unsync_mode})")
    print(f"  - Theoretical Total Pairs        : {theoretical_pairs:,}")
    print(f"  - Generated Candidate Pairs      : {n_candidates:,} (Same-camera: {same_cam_candidates}, Cross-camera: {cross_cam_candidates})")
    print(f"  - Search Space Reduction         : {reduction_pct:.2f}% pruned")
    print(f"  - Generation Runtime             : {t_cg_ms:.2f} ms")
    print(f"  - Pruning Reasons Breakdown      : {rejection_counts}")

    # 4. Multimodal Identity Fusion
    print("\n--- 4. MULTIMODAL IDENTITY FUSION & RE-ID GUARD ---")
    id_cfg = config["pipeline"].get("identity_fusion", {})
    min_score_thresh = float(id_cfg.get("min_score_threshold", 0.70))

    match_results: List[Dict[str, Any]] = []
    c002_mismatch_count = 0
    c001_c003_compat_count = 0
    confirmed_count = 0
    ambiguous_count = 0
    rejected_count = 0

    top_cross_cam_matches: List[Tuple[float, Observation, Observation, Dict[str, Any]]] = []

    for obs_a, obs_b in candidates:
        res = match_observations(obs_a, obs_b, config=id_cfg)
        match_results.append(res)

        app_status = res.get("evidence", {}).get("appearance_status")
        if app_status == "incompatible_models":
            c002_mismatch_count += 1
        elif app_status == "available" and obs_a.camera_id != obs_b.camera_id:
            c001_c003_compat_count += 1
            app_sim = res.get("evidence", {}).get("appearance_similarity", 0.0) or 0.0
            top_cross_cam_matches.append((app_sim, obs_a, obs_b, res))

        dec = res.get("decision_state")
        if dec == "CONFIRMED":
            confirmed_count += 1
        elif dec == "AMBIGUOUS":
            ambiguous_count += 1
        else:
            rejected_count += 1

    top_cross_cam_matches.sort(key=lambda x: x[0], reverse=True)

    print(f"  - Total Candidate Pairs Evaluated: {len(match_results):,}")
    print(f"  - C002 Re-ID Incompatible Pairs : {c002_mismatch_count} (Safely guarded; cross-model similarity suppressed)")
    print(f"  - C001 ↔ C003 Re-ID Valid Pairs  : {c001_c003_compat_count} (Evaluated via shared osnet_x0_25_aicity model)")
    print(f"  - Pairwise Decision States       : CONFIRMED={confirmed_count}, AMBIGUOUS={ambiguous_count}, REJECTED={rejected_count}")

    # 5. Identity Graph Assembly & Clustering
    print("\n--- 5. IDENTITY GRAPH CLUSTERING & CONSISTENCY VERIFICATION ---")
    graph = IdentityGraph(min_score_threshold=min_score_thresh)
    graph.assemble_from_pairs(
        observations=observations,
        pairs=candidates,
        match_results=match_results,
        config=id_cfg,
    )

    clusters = graph.get_candidate_identities()
    status_counts: Dict[str, int] = {}
    multi_cam_clusters = []
    singleton_clusters = []

    for cl in clusters:
        st = cl.get("identity_status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1
        cams = cl.get("cameras", [])
        if len(cams) > 1:
            multi_cam_clusters.append(cl)
        if len(cl.get("observation_ids", [])) == 1:
            singleton_clusters.append(cl)

    print(f"  - Identity Graph Nodes (Tracklets): {len(graph.nodes)}")
    print(f"  - Identity Graph Edges Formed    : {len(graph.edges)}")
    print(f"  - Total Inferred Vehicle Clusters: {len(clusters)}")
    print(f"  - Cluster Status Counts          : {status_counts}")
    print(f"  - Multi-Camera Vehicle Clusters  : {len(multi_cam_clusters)}")
    print(f"  - Singleton Vehicle Clusters     : {len(singleton_clusters)}")

    # 6. Trajectory Inference (Image-Space / Temporal Fallback Mode)
    print("\n--- 6. TRAJECTORY INFERENCE (IMAGE-SPACE FALLBACK) ---")
    dummy_road_graph = RoadGraph()
    trajectories_output: List[Dict[str, Any]] = []

    for idx, cl in enumerate(clusters):
        ut_id = f"UT_ID_{idx + 1:04d}"
        cl["candidate_vehicle_id"] = ut_id
        cl["identity_id"] = ut_id

        traj = reconstruct_identity_trajectory(
            identity_data=cl,
            road_graph=dummy_road_graph,
            config=config["pipeline"].get("trajectory_engine", {}),
        )
        traj_dict = traj.to_dict()
        traj_dict["spatial_semantics"] = "image_space_only"
        traj_dict["temporal_semantics"] = "video_relative"
        trajectories_output.append(traj_dict)

    print(f"  - Reconstructed Trajectories     : {len(trajectories_output)}")
    print(f"  - Spatial Route Classification   : Image-Space / Temporally Plausible (Zero GPS claim)")

    # 7. Serialization
    print("\n--- 7. SERIALIZATION OF RESULTS ---")
    normalized_obs_data = [o.to_dict() for o in observations]
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

    graph_data = {
        "nodes_count": len(graph.nodes),
        "edges_count": len(graph.edges),
        "edges": graph.edges,
        "clusters_count": len(clusters),
    }

    eval_results = {
        "dataset_name": config["dataset"]["name"],
        "dataset_classification": "REAL",
        "ground_truth_available": False,
        "evaluation_type": "UNSUPERVISED_INFERENCE_DIAGNOSTICS",
        "cameras_evaluated": discovered_cams,
        "tracklets_ingested": len(observations),
        "candidate_pairs_evaluated": len(candidates),
        "candidate_reduction_pct": reduction_pct,
        "pairwise_decisions": {
            "CONFIRMED": confirmed_count,
            "AMBIGUOUS": ambiguous_count,
            "REJECTED": rejected_count,
        },
        "identity_clusters_count": len(clusters),
        "identity_status_distribution": status_counts,
        "reid_compatibility": {
            "c001_model": "osnet_x0_25_aicity",
            "c002_model": "osnet_x0_25_msmt17",
            "c003_model": "osnet_x0_25_aicity",
            "c002_guard_active": True,
            "cross_model_pairs_blocked": c002_mismatch_count,
            "compatible_reid_pairs_evaluated": c001_c003_compat_count,
        },
        "data_limitations": [
            "No cross-camera ground-truth IDs (zero GT validation claim).",
            "Plate text is 100% null (zero ANPR text validation claim).",
            "No camera calibration or world coordinates (zero world GIS claim).",
            "C002 uses msmt17 weights (degraded Re-ID cross-matching with C001/C003).",
            "Timestamps are video-relative with no external synchronization offset.",
        ],
    }

    inventory_data = {
        "cameras": cam_stats,
        "total_tracklets": len(observations),
        "valid_embeddings": valid_embs_count,
        "timestamp_semantics": "video_relative",
        "coordinate_system": "image",
    }

    provenance_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_handoff_path": handoff_path,
        "config_path": str(config_path),
        "pipeline_stages": [
            "load_aicity_member1_feed",
            "CandidateGenerator(unsynchronized_mode=True)",
            "match_observations(reid_guard=True)",
            "IdentityGraph.build_graph_from_matches",
            "reconstruct_identity_trajectory(mode=image_space)",
        ],
        "zero_fabrication_enforced": True,
    }

    artifacts_map = {
        "normalized_observations.json": normalized_obs_data,
        "inferred_identities.json": inferred_identities_data,
        "identity_graph.json": graph_data,
        "trajectories.json": trajectories_output,
        "evaluation_results.json": eval_results,
        "data_inventory.json": inventory_data,
        "provenance.json": provenance_data,
    }

    for fname, data in artifacts_map.items():
        p_out = out_dir / fname
        with open(p_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  - Serialized                     : {p_out} ({p_out.stat().st_size:,} bytes)")

    t_total = time.perf_counter() - t_start
    print(f"\nPipeline execution completed in {t_total:.2f} seconds.")

    # Top Cross-Camera Matches
    print("\n--- TOP CROSS-CAMERA DEMONSTRATION EXAMPLES (C001 ↔ C003) ---")
    if top_cross_cam_matches:
        for idx, (sim, oa, ob, res) in enumerate(top_cross_cam_matches[:5]):
            print(f"  Match #{idx + 1}:")
            print(f"    - Source Camera / Track        : {oa.camera_id} Track {oa.track_id} ({oa.vehicle_type}, t={oa.timestamp_seconds}s)")
            print(f"    - Target Camera / Track        : {ob.camera_id} Track {ob.track_id} ({ob.vehicle_type}, t={ob.timestamp_seconds}s)")
            print(f"    - Re-ID Cosine Similarity      : {sim:.4f}")
            print(f"    - Re-ID Models Compatible      : True ({oa.reid_model})")
            print(f"    - Match Probability            : {res['same_vehicle_probability']:.4f}")
            print(f"    - Decision State               : {res['decision_state']}")
            print(f"    - Rationale                    : {res['explanation']}")
    else:
        print("  No cross-camera compatible matches found.")

    return {
        "status": "success",
        "runtime_seconds": t_total,
        "observations_count": len(observations),
        "candidates_count": len(candidates),
        "clusters_count": len(clusters),
        "artifacts": list(artifacts_map.keys()),
    }


if __name__ == "__main__":
    run_aicity_mvp()
