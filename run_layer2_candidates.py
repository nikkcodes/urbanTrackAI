"""
UrbanTrack AI — Layer 2 Candidate Generation Runner Script.

Executes production candidate generation over the frozen canonical ingestion output:
results/layer2_ingestion/canonical_tracklets.json

Generates:
- results/layer2_candidates/candidate_pairs.json
- results/layer2_candidates/candidate_summary.json
- results/layer2_candidates/candidate_rejections.json
- results/layer2_candidates/candidate_validation.json
"""

import sys
import time
from pathlib import Path

from layer2.candidate_generation.candidate_generator import CandidateGenerator


def main():
    print("=" * 80)
    print("UrbanTrack AI — Layer 2 Candidate Generation Pipeline")
    print("=" * 80)

    input_json = Path("results/layer2_ingestion/canonical_tracklets.json")
    output_dir = Path("results/layer2_candidates")

    if not input_json.is_file():
        print(f"[ERROR] Canonical tracklets not found at {input_json}")
        sys.exit(1)

    generator = CandidateGenerator(
        graph_path="UrbanTrack_Member1_Handoff 2/data/config/camera_graph.json",
        locations_path="UrbanTrack_Member1_Handoff 2/data/config/camera_locations.json",
        max_speed_mps=45.0,
        max_time_seconds=None,
    )

    print(f"Loaded camera graph ({generator.topology_gate.edge_count} directed edges)")
    print(f"Loaded camera locations ({len(generator.spatial_gate._locations)} cameras)")
    print(f"Processing canonical tracklets from: {input_json}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    summary = generator.execute_and_serialize(
        tracklets_json_path=input_json,
        output_dir=output_dir,
    )

    print("\nExecution Completed Successfully!")
    print(f"Runtime: {summary['execution_metadata']['runtime_seconds']:.2f}s")
    print(f"Peak Memory: {summary['execution_metadata']['peak_memory_mb']} MB")
    print("-" * 80)
    print("Candidate Generation Metrics:")
    scale = summary["dataset_scale"]
    metrics = summary["candidate_generation_metrics"]
    print(f"  Total Tracklets:                    {scale['total_tracklets']:,}")
    print(f"  Theoretical Global Pairs:           {scale['theoretical_global_pairs']:,}")
    print(f"  Theoretical Intra-Scenario Pairs:   {scale['theoretical_intra_scenario_pairs']:,}")
    print(f"  Cross-Scenario Eliminated:          {scale['cross_scenario_pairs_eliminated']:,}")
    print(f"  Intra-Scenario Rejected:            {metrics['intra_scenario_pairs_rejected']:,}")
    print(f"  Candidate Pairs Generated:          {metrics['candidate_pairs_generated']:,}")
    print(f"  Reduction vs Global Pairs:          {metrics['reduction_percentage_vs_global']:.2f}%")
    print(f"  Reduction vs Intra-Scenario Pairs:  {metrics['reduction_percentage_vs_intra_scenario']:.2f}%")
    print("\nRejection Breakdown:")
    for reason, count in sorted(metrics["rejection_breakdown"].items()):
        print(f"  {reason:30s}: {count:,}")
    print("\nScenario Breakdown:")
    for sc, sc_data in sorted(summary["scenario_breakdown"].items()):
        print(
            f"  {sc}: {sc_data['tracklet_count']:4d} tracklets | "
            f"Theoretical: {sc_data['theoretical_pairs']:9,d} | "
            f"Candidates: {sc_data['candidate_pairs']:9,d} | "
            f"Rejected: {sc_data['rejected_pairs']:7,d} | "
            f"Reduction: {sc_data['reduction_percentage']:5.2f}% | "
            f"Time: {sc_data['runtime_seconds']:.2f}s"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
