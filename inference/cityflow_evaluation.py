"""Executable evaluation for the extracted CityFlowV2 S01 benchmark.

The evaluator deliberately keeps annotation labels outside Observation objects.
It evaluates camera-local MTSC track summaries, which is the appropriate unit
for cross-camera identity matching when the input contains no Re-ID vectors or
license plates.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from schemas.observation_schema import Observation
from .cityflow_adapter import CityFlowV2Adapter
from .candidate_generation import CandidateGenerator


def _iou(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> float:
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def summarize_tracklets(observations: List[Observation]) -> Tuple[List[Observation], Dict[str, Dict[str, Any]]]:
    """Create one representative Observation per camera-local track.

    The complete frame observations remain the source evidence.  Summaries are
    used only to make the cross-camera identity benchmark computationally
    tractable and retain track history/provenance for auditability.
    """
    grouped: Dict[Tuple[str, str], List[Observation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.camera_id, str(obs.local_track_id or obs.track_id))].append(obs)

    summaries: List[Observation] = []
    index: Dict[str, Dict[str, Any]] = {}
    for (camera_id, local_track_id), members in sorted(grouped.items()):
        members.sort(key=lambda item: (item.timestamp_seconds, item.observation_id))
        anchor = members[len(members) // 2]
        positions = [item.world_position for item in members if item.world_position is not None]
        world_position = None
        if positions:
            world_position = [
                sum(float(pos[0]) for pos in positions) / len(positions),
                sum(float(pos[1]) for pos in positions) / len(positions),
            ]
        history = [
            {
                "frame_number": item.frame_number,
                "timestamp_seconds": item.timestamp_seconds,
                "bbox": item.bbox,
                "world_position": item.world_position,
            }
            for item in members
        ]
        summary_id = f"CF_TRACKLET_{camera_id}_{local_track_id}"
        summary = Observation(
            observation_id=summary_id,
            camera_id=camera_id,
            frame_id=anchor.frame_id,
            frame_number=anchor.frame_number,
            timestamp_seconds=anchor.timestamp_seconds,
            track_id=local_track_id,
            local_track_id=local_track_id,
            bbox=anchor.bbox,
            trajectory_point=anchor.trajectory_point,
            point_coordinate_system=anchor.point_coordinate_system,
            world_position=world_position,
            world_coordinate_system=anchor.world_coordinate_system,
            detection_confidence=(
                sum(float(item.detection_confidence) for item in members if item.detection_confidence is not None)
                / max(1, sum(1 for item in members if item.detection_confidence is not None))
                if any(item.detection_confidence is not None for item in members) else None
            ),
            timestamp_semantics=anchor.timestamp_semantics,
            time_reference_id=anchor.time_reference_id,
            source_dataset=anchor.source_dataset,
            source_video=anchor.source_video,
            local_track_history=history,
            data_quality_flags=sorted({flag for item in members for flag in item.data_quality_flags}),
            source_provenance={
                **(anchor.source_provenance or {}),
                "tracklet_summary": True,
                "source_observation_count": len(members),
                "source_first_frame": members[0].frame_number,
                "source_last_frame": members[-1].frame_number,
            },
        )
        summaries.append(summary)
        index[summary_id] = {
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "source_observations": members,
        }
    return summaries, index


def _assign_ground_truth(
    summaries: List[Observation],
    summary_index: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float = 0.50,
) -> Dict[str, Optional[int]]:
    """Assign labels for evaluation only by frame-local box matching."""
    labels: Dict[str, Optional[int]] = {}
    for summary in summaries:
        camera_gt = ground_truth.get(f"S01/{summary.camera_id}", [])
        matches: List[int] = []
        for source_obs in summary_index[summary.observation_id]["source_observations"]:
            best_id = None
            best_iou = 0.0
            for gt in camera_gt:
                if int(gt["frame"]) != int(source_obs.frame_number):
                    continue
                candidate_box = gt.get("bbox")
                score = _iou(source_obs.bbox, candidate_box)
                if score > best_iou:
                    best_iou = score
                    best_id = int(gt["track_id"])
            if best_id is not None and best_iou >= iou_threshold:
                matches.append(best_id)
        labels[summary.observation_id] = Counter(matches).most_common(1)[0][0] if matches else None
    return labels


def _pair_metrics(
    summaries: List[Observation],
    labels: Dict[str, Optional[int]],
    predicted_pairs: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    true_pairs: Set[Tuple[str, str]] = set()
    all_cross_camera_pairs: Set[Tuple[str, str]] = set()
    for index, left in enumerate(summaries):
        for right in summaries[index + 1:]:
            if left.camera_id == right.camera_id:
                continue
            pair = tuple(sorted((left.observation_id, right.observation_id)))
            all_cross_camera_pairs.add(pair)
            if labels.get(left.observation_id) is not None and labels.get(left.observation_id) == labels.get(right.observation_id):
                true_pairs.add(pair)
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    tn = len(all_cross_camera_pairs - predicted_pairs - true_pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_merge = fp / (fp + tn) if fp + tn else 0.0
    return {
        "cross_camera_pairs": len(all_cross_camera_pairs),
        "labeled_tracklets": sum(value is not None for value in labels.values()),
        "positive_pairs": len(true_pairs),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merge_rate": round(false_merge, 4),
        "false_split_rate": round(fn / len(true_pairs), 4) if true_pairs else 0.0,
    }


def run_cityflow_s01_evaluation(
    project_root: Optional[Path] = None,
    tracker_filename: str = "mtsc_deepsort_mask_rcnn.txt",
) -> Dict[str, Any]:
    """Run CityFlow S01 evaluation from extracted native files."""
    root = project_root or Path(__file__).resolve().parent.parent
    dataset_root = root / "data" / "cityflowv2"
    adapter = CityFlowV2Adapter(dataset_root=dataset_root)
    frame_observations, ground_truth = adapter.load_scenario_from_directory(
        split="train", scenario_id="S01", tracker_filename=tracker_filename
    )
    summaries, summary_index = summarize_tracklets(frame_observations)
    labels = _assign_ground_truth(summaries, summary_index, ground_truth)

    candidate_generator = CandidateGenerator(min_score_threshold=0.70)
    candidate_list, rejection_counts = candidate_generator.generate_candidates(summaries)
    candidate_ids = {
        tuple(sorted((left.observation_id, right.observation_id)))
        for left, right in candidate_list
    }
    predicted_pairs: Set[Tuple[str, str]] = set()

    candidate_pairs = 0
    positive_candidates = 0
    positive_pairs = 0
    for index, left in enumerate(summaries):
        for right in summaries[index + 1:]:
            if left.camera_id == right.camera_id:
                continue
            pair = tuple(sorted((left.observation_id, right.observation_id)))
            is_positive = labels.get(left.observation_id) is not None and labels.get(left.observation_id) == labels.get(right.observation_id)
            positive_pairs += int(is_positive)
            if pair in candidate_ids:
                candidate_pairs += 1
                positive_candidates += int(is_positive)

    metrics = _pair_metrics(summaries, labels, predicted_pairs)
    return {
        "metadata": {
            "dataset": "AI City Challenge 2022 CityFlowV2",
            "split": "train",
            "scenario": "S01",
            "tracker_input": tracker_filename,
            "source_root": "data/cityflowv2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ground_truth_used_for": "evaluation_only_box_matching",
            "ground_truth_inference_leakage": False,
            "source_videos_extracted": False,
        },
        "ingestion": {
            "frame_observations": len(frame_observations),
            "tracklet_summaries": len(summaries),
            "cameras": sorted({obs.camera_id for obs in summaries}),
            "observations_with_512d_embeddings": sum(
                bool(obs.appearance_embedding and len(obs.appearance_embedding) == 512) for obs in frame_observations
            ),
            "observations_with_plates": sum(obs.plate is not None for obs in frame_observations),
            "observations_with_native_world_position": sum(obs.world_position is not None for obs in frame_observations),
        },
        "identity": {
            "metrics": metrics,
            "predicted_identity_edges": len(predicted_pairs),
            "candidate_pairs": candidate_pairs,
            "candidate_positive_recall": round(positive_candidates / positive_pairs, 4) if positive_pairs else None,
            "candidate_rejection_summary": rejection_counts,
            "interpretation": "No identity edges are confirmed when CityFlow MTSC input lacks plate or appearance evidence.",
        },
    }


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    report = run_cityflow_s01_evaluation(project_root=project_root)
    output_path = project_root / "results" / "cityflow_s01_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
