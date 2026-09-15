"""
UrbanTrack AI — Multi-Camera Benchmark Runner.
Executes end-to-end benchmark generation, inference execution, and evaluation.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.observation_schema import Observation
from inference.observation_loader import load_observations_from_json
from .generator import MultiCameraBenchmarkGenerator
from .ground_truth import GroundTruthRegistry
from .evaluator import BenchmarkEvaluationResult, MultiCameraBenchmarkEvaluator


def run_multicamera_benchmark(
    data_dir: Optional[Path] = None,
    benchmark_id: str = "multicamera_v1",
    n_vehicles: int = 150,
    target_observations: int = 1500,
    threshold: float = 0.75,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """
    Programmatic entrypoint to run the independent multi-camera benchmark.

    Args:
        data_dir: Path to directory storing multicamera_v1 benchmark assets.
        n_vehicles: Number of latent vehicles if generating fresh data.
        target_observations: Number of observations to synthesize.
        threshold: Operating decision threshold for confirmed matches.
        force_regenerate: Whether to overwrite existing benchmark files.

    Returns:
        Dict[str, Any]: Complete benchmark evaluation report.
    """
    bench_dir = data_dir or (PROJECT_ROOT / "data" / "benchmarks" / benchmark_id)
    bench_dir.mkdir(parents=True, exist_ok=True)

    obs_file = bench_dir / "observations.json"
    gt_file = bench_dir / "ground_truth.json"
    cam_file = bench_dir / "cameras.json"
    pw_file = bench_dir / "pairwise_ground_truth.json"

    # 1. Check if dataset needs generation
    if force_regenerate or not (obs_file.exists() and gt_file.exists() and pw_file.exists() and cam_file.exists()):
        generator = MultiCameraBenchmarkGenerator(seed=42, project_root=PROJECT_ROOT)
        generator.export_to_directory(
            output_dir=bench_dir,
            n_vehicles=n_vehicles,
            target_observations=target_observations,
            n_hard_negatives=20,
        )

    # 2. Load cameras
    with open(cam_file, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    # 3. Load independent ground truth registry
    registry = GroundTruthRegistry.load_from_directory(bench_dir)

    # 4. Load observations
    observations = load_observations_from_json(obs_file)

    # 5. Evaluate production reasoning pipeline
    evaluator = MultiCameraBenchmarkEvaluator(
        registry=registry,
        camera_metadata=cameras,
        threshold=threshold,
    )
    result = evaluator.evaluate(observations)

    return result.to_dict()


def main() -> None:
    print("=" * 80)
    print(" URBANTRACK AI — INDEPENDENT MULTI-CAMERA BENCHMARK RUNNER ")
    print("=" * 80)

    bench_dir = PROJECT_ROOT / "data" / "benchmarks" / "multicamera_v1"
    res = run_multicamera_benchmark(data_dir=bench_dir, force_regenerate=True)

    print(f"\nDataset: {res['dataset_name']}")
    print(f"Total Observations: {res['total_observations']}")
    print(f"Total Theoretical Pairs: {res['total_possible_pairs']:,}")
    print(f"Candidate Pairs Retained: {res['candidate_pairs_count']:,} ({res['candidate_reduction_pct']}% reduction)")
    print(f"Candidate Recall: {res['candidate_recall_pct']}%")
    print(f"\n--- PAIRWISE PERFORMANCE ---")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall: {res['recall']:.4f}")
    print(f"F1 Score: {res['f1_score']:.4f}")
    print(f"False Merge Rate (FMR): {res['false_merge_rate']:.4f}")
    print(f"False Split Rate (FSR): {res['false_split_rate']:.4f}")
    print(f"Ambiguity Rate: {res['ambiguity_rate']:.4f}")
    print(f"Discovered Clusters: {res['cluster_count']} (Purity: {res['cluster_purity']:.4f})")
    print(f"\n--- HARD NEGATIVE SAFETY ---")
    print(f"Hard Negatives Evaluated: {res['hard_negatives_evaluated']}")
    print(f"Hard Negative False Merges: {res['hard_negative_false_merges']}")
    print(f"Hard Negative Safe Rejection Rate: {res['hard_negative_safe_rate']}%")

    print(f"\n--- DIFFICULTY TIER BREAKDOWN ---")
    for tier, stats in res["tier_breakdown"].items():
        print(f"  [{tier:11s}] Pairs: {stats['total_pairs']:4d} | Prec: {stats['precision']:.4f} | Rec: {stats['recall']:.4f} | F1: {stats['f1_score']:.4f} | FMR: {stats['false_merge_rate']:.4f}")

    print(f"\nTotal Runtime: {res['elapsed_total_seconds']:.2f}s (Candidate: {res['elapsed_candidate_gen_seconds']:.2f}s, Fusion: {res['elapsed_fusion_seconds']:.2f}s, Graph: {res['elapsed_graph_seconds']:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
