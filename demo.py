"""
Day 2 End-to-End Demo Script for UrbanTrack AI Identity Fusion Engine.
Demonstrates observation loading, pairwise evidence calculation, estimated match probabilities,
identity graph generation, and candidate vehicle clustering.
"""

from pathlib import Path

from inference import (
    IdentityGraph,
    load_camera_metadata,
    load_observations_from_json,
    match_observations,
)


def run_demo() -> None:
    print("==================================================================")
    print("      UrbanTrack AI - Mobility Inference Engine (Day 2 Demo)      ")
    print("==================================================================\n")

    base_dir = Path(__file__).parent
    cam_meta_file = base_dir / "data" / "cameras" / "camera_metadata.json"
    obs_feed_file = base_dir / "data" / "observations" / "sample_feed.json"

    # 1. Load Camera Metadata
    print(f"[1] Loading camera metadata from: {cam_meta_file.name}")
    camera_metadata = load_camera_metadata(cam_meta_file)
    for cam_id, meta in camera_metadata.items():
        print(f"    - {cam_id}: GPS ({meta['latitude']}, {meta['longitude']}) - {meta.get('name', 'N/A')}")

    # 2. Load Synthetic Observations Feed
    print(f"\n[2] Loading observations from: {obs_feed_file.name}")
    observations = load_observations_from_json(obs_feed_file, camera_metadata=camera_metadata)
    for obs in observations:
        print(f"    - [{obs.observation_id}] Camera: {obs.camera_id} | Type: {obs.vehicle_type} | Time: {obs.timestamp_seconds}s | GPS: ({obs.latitude}, {obs.longitude})")

    # 3. Perform Pairwise Identity Fusion
    print("\n[3] Evaluating Pairwise Identity Fusion (Observation A vs Observation B)")
    obs_a = observations[0]  # cam_01 (car)
    obs_b = observations[1]  # cam_02 (car, same vehicle)
    obs_c = observations[2]  # cam_03 (bus, different vehicle type)

    print("\n--- Match Evaluation: Observation A vs Observation B (Same Car) ---")
    res_ab = match_observations(obs_a, obs_b, camera_metadata=camera_metadata)
    print(f"Same Vehicle Probability : {res_ab['same_vehicle_probability']}")
    print("Evidence Breakdown:")
    for k, v in res_ab['evidence'].items():
        print(f"  - {k:28s}: {v}")
    print(f"Explanation              : {res_ab['explanation']}")

    print("\n--- Match Evaluation: Observation A vs Observation C (Car vs Bus) ---")
    res_ac = match_observations(obs_a, obs_c, camera_metadata=camera_metadata)
    print(f"Same Vehicle Probability : {res_ac['same_vehicle_probability']}")
    print("Evidence Breakdown:")
    for k, v in res_ac['evidence'].items():
        print(f"  - {k:28s}: {v}")
    print(f"Explanation              : {res_ac['explanation']}")

    # 4. Construct Identity Graph & Cluster Candidate Identities
    print("\n[4] Building Identity Graph & Grouping Candidate Vehicle Identities")
    graph = IdentityGraph(min_probability_threshold=0.5)
    graph.build_graph(observations, camera_metadata=camera_metadata)

    candidates = graph.get_candidate_identities()
    print(f"\nDiscovered {len(candidates)} Candidate Vehicle Identities:")
    for cand in candidates:
        print(f"\n  Identity ID   : {cand['candidate_vehicle_id']}")
        print(f"  Observations  : {cand['observation_ids']}")
        print(f"  Cameras Trail : {' -> '.join(cand['cameras_visited'])}")
        print(f"  Total Members : {cand['member_observations_count']}")

    print("\n==================================================================")
    print("                 Day 2 End-to-End Demo Complete                   ")
    print("==================================================================")


if __name__ == "__main__":
    run_demo()
