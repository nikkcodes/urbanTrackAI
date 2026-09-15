"""Synthetic perception degradation generator for benchmarking.

This module is ONLY for benchmarking. It reads the real perception outputs
(``observations.json`` and ``trajectories.json``) and produces degraded
copies in a separate directory (``data/synthetic_output/``). It never
modifies or overwrites the real perception outputs, and it never fabricates
detections, plates, embeddings, or GPS values -- it only removes or weakens
existing observed information.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


class SyntheticDegradationGenerator:
    """Generate degraded copies of real perception observations."""

    def __init__(
        self,
        seed: int | None = None,
        drop_frame_rate: float = config.SYNTHETIC_DROP_FRAME_RATE,
        missing_plate_rate: float = config.SYNTHETIC_MISSING_PLATE_RATE,
        ocr_failure_rate: float = config.SYNTHETIC_OCR_FAILURE_RATE,
        low_ocr_confidence_rate: float = config.SYNTHETIC_LOW_OCR_RATE,
        missing_reid_rate: float = config.SYNTHETIC_MISSING_REID_RATE,
        occlusion_rate: float = config.SYNTHETIC_OCCLUSION_RATE,
        confidence_degradation_rate: float = config.SYNTHETIC_CONFIDENCE_DEGRADATION_RATE,
        track_fragment_rate: float = config.SYNTHETIC_TRACK_FRAGMENT_RATE,
        camera_outage_rate: float = config.SYNTHETIC_CAMERA_OUTAGE_RATE,
    ) -> None:
        """Store degradation probabilities and seed the RNG."""
        self.seed = seed
        self._rng = random.Random(seed)
        self.drop_frame_rate = self._clamp_prob(drop_frame_rate)
        self.missing_plate_rate = self._clamp_prob(missing_plate_rate)
        self.ocr_failure_rate = self._clamp_prob(ocr_failure_rate)
        self.low_ocr_confidence_rate = self._clamp_prob(low_ocr_confidence_rate)
        self.missing_reid_rate = self._clamp_prob(missing_reid_rate)
        self.occlusion_rate = self._clamp_prob(occlusion_rate)
        self.confidence_degradation_rate = self._clamp_prob(
            confidence_degradation_rate
        )
        self.track_fragment_rate = self._clamp_prob(track_fragment_rate)
        self.camera_outage_rate = self._clamp_prob(camera_outage_rate)

    @staticmethod
    def _clamp_prob(value: float) -> float:
        """Clamp a probability into the inclusive [0, 1] range."""
        return max(0.0, min(1.0, float(value)))

    def generate(
        self,
        input_dir: str | Path = config.SYNTHETIC_INPUT_DIR,
        output_dir: str | Path = config.SYNTHETIC_OUTPUT_DIR,
    ) -> dict[str, Any]:
        """Generate degraded copies and return the degradation summary."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        observations = self._load_json(input_path / "observations.json")
        trajectories = self._load_json(input_path / "trajectories.json")

        degraded_observations, obs_summary = self._degrade_observations(observations)
        degraded_trajectories, traj_summary = self._degrade_trajectories(trajectories)

        summary = {
            "source_dataset": str(input_path),
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "degradation_seed": self.seed,
            "degraded_frames": obs_summary["degraded_frames"],
            "degraded_tracks": traj_summary["degraded_tracks"],
            "missing_plate_count": obs_summary["missing_plate_count"],
            "ocr_failure_count": obs_summary["ocr_failure_count"],
            "missing_embedding_count": traj_summary["missing_embedding_count"],
            "occluded_vehicle_count": obs_summary["occluded_vehicle_count"],
            "fragmented_track_count": traj_summary["fragmented_track_count"],
            "dropped_frame_count": obs_summary["dropped_frame_count"],
            "camera_outage_intervals": obs_summary["camera_outage_intervals"],
        }

        self._write_json(output_path / "observations_degraded.json", degraded_observations)
        self._write_json(output_path / "trajectories_degraded.json", degraded_trajectories)
        self._write_json(output_path / "degradation_summary.json", summary)
        return summary

    # ------------------------------------------------------------------
    # Observation degradation
    # ------------------------------------------------------------------
    def _degrade_observations(
        self, observations: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return degraded observations plus a per-observation summary."""
        frames = observations.get("frames", []) if isinstance(observations, dict) else []
        degraded_frames: list[dict[str, Any]] = []
        summary = {
            "degraded_frames": 0,
            "missing_plate_count": 0,
            "ocr_failure_count": 0,
            "occluded_vehicle_count": 0,
            "dropped_frame_count": 0,
            "camera_outage_intervals": [],
        }
        frame_numbers = [
            int(frame.get("frame_number"))
            for frame in frames
            if isinstance(frame, dict) and "frame_number" in frame
        ]
        outage_ranges = self._camera_outage_ranges(frame_numbers)
        for start, end in outage_ranges:
            summary["camera_outage_intervals"].append([start, end])

        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_number = int(frame.get("frame_number", -1))
            if self._in_outage(frame_number, outage_ranges):
                continue
            if self._rng.random() < self.drop_frame_rate:
                summary["dropped_frame_count"] += 1
                continue

            degraded_frame = self._degrade_frame(frame, frame_number, summary)
            if degraded_frame is not None:
                degraded_frames.append(degraded_frame)
                summary["degraded_frames"] += 1

        observations_out = dict(observations) if isinstance(observations, dict) else {}
        observations_out["frames"] = degraded_frames
        summary["fragmented_track_count"] = len(summary.pop("fragmented_tracks", set()))
        return observations_out, summary

    def _degrade_frame(
        self,
        frame: dict[str, Any],
        frame_number: int,
        summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a degraded copy of a single frame observation."""
        degraded = dict(frame)
        vehicles = frame.get("vehicles", [])
        if not isinstance(vehicles, list):
            degraded["vehicles"] = []
            degraded["synthetic"] = True
            degraded["degradation_tags"] = ["dropped_frame"]
            return degraded

        degraded_vehicles: list[dict[str, Any]] = []
        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                continue
            degraded_vehicle = dict(vehicle)
            tags: list[str] = []

            track_id = int(degraded_vehicle.get("track_id", -1))
            if self._rng.random() < self.track_fragment_rate:
                summary.setdefault("fragmented_tracks", set()).add(track_id)
                tags.append("track_fragmentation")

            if self._rng.random() < self.missing_plate_rate:
                degraded_vehicle["plate_number"] = None
                degraded_vehicle["plate_text_confidence"] = None
                tags.append("missing_plate")
                summary["missing_plate_count"] += 1
            elif self._rng.random() < self.ocr_failure_rate:
                degraded_vehicle["plate_number"] = None
                degraded_vehicle["ocr_confidence"] = None
                tags.append("ocr_failure")
                summary["ocr_failure_count"] += 1
            elif self._rng.random() < self.low_ocr_confidence_rate:
                confidence = degraded_vehicle.get("plate_text_confidence")
                if isinstance(confidence, (int, float)) and confidence > 0:
                    degraded_vehicle["plate_text_confidence"] = round(
                        float(confidence) * self._rng.uniform(0.2, 0.6), 4
                    )
                    tags.append("low_ocr_confidence")

            if self._rng.random() < self.occlusion_rate:
                degraded_vehicle["occluded"] = True
                degraded_vehicle["confidence"] = round(
                    self._rng.uniform(
                        config.SYNTHETIC_OCCLUSION_CONFIDENCE_MIN,
                        config.SYNTHETIC_OCCLUSION_CONFIDENCE_MAX,
                    ),
                    4,
                )
                tags.append("occlusion")
                summary["occluded_vehicle_count"] += 1
            elif self._rng.random() < self.confidence_degradation_rate:
                confidence = degraded_vehicle.get("confidence")
                if isinstance(confidence, (int, float)) and confidence > 0:
                    degraded_vehicle["confidence"] = round(
                        max(
                            config.SYNTHETIC_CONFIDENCE_DEGRADATION_MIN,
                            float(confidence) * self._rng.uniform(0.4, 0.85),
                        ),
                        4,
                    )
                    tags.append("confidence_degradation")

            if tags:
                degraded_vehicle["synthetic"] = True
                degraded_vehicle["degradation_tags"] = tags
            degraded_vehicles.append(degraded_vehicle)

        degraded["vehicles"] = degraded_vehicles
        degraded["synthetic"] = True
        degraded["degradation_tags"] = sorted(
            {tag for vehicle in degraded_vehicles for tag in vehicle.get("degradation_tags", [])}
        )
        return degraded

    def _camera_outage_ranges(
        self, frame_numbers: list[int]
    ) -> list[tuple[int, int]]:
        """Return ``[start_frame, end_frame]`` outage intervals."""
        if not frame_numbers or self.camera_outage_rate <= 0:
            return []
        frame_numbers = sorted(set(frame_numbers))
        outages: list[tuple[int, int]] = []
        max_window = max(1, len(frame_numbers) // 4)
        index = 0
        while index < len(frame_numbers):
            if self._rng.random() < self.camera_outage_rate:
                end_index = min(len(frame_numbers), index + max_window)
                outages.append((frame_numbers[index], frame_numbers[end_index - 1]))
                index = end_index
            else:
                index += 1
        return outages

    @staticmethod
    def _in_outage(frame_number: int, outages: list[tuple[int, int]]) -> bool:
        """Return whether a frame falls inside any outage interval."""
        return any(start <= frame_number <= end for start, end in outages)

    # ------------------------------------------------------------------
    # Trajectory degradation
    # ------------------------------------------------------------------
    def _degrade_trajectories(
        self, trajectories: list[Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return degraded trajectories plus a per-track summary."""
        summary = {
            "degraded_tracks": 0,
            "missing_embedding_count": 0,
            "fragmented_track_count": 0,
        }
        if not isinstance(trajectories, list):
            return [], summary

        degraded: list[dict[str, Any]] = []
        for record in trajectories:
            if not isinstance(record, dict):
                continue
            degraded_record = dict(record)
            tags: list[str] = []
            track_id = int(degraded_record.get("track_id", -1))

            if self._rng.random() < self.missing_reid_rate:
                degraded_record["appearance_embedding"] = None
                degraded_record["embedding_quality"] = None
                tags.append("missing_reid_embedding")
                summary["missing_embedding_count"] += 1

            if self._rng.random() < self.track_fragment_rate:
                trajectory = degraded_record.get("trajectory")
                if isinstance(trajectory, list) and len(trajectory) > 2:
                    keep = max(1, len(trajectory) // 2)
                    degraded_record["trajectory"] = trajectory[:keep]
                    tags.append("track_fragmentation")
                summary["fragmented_track_count"] += 1

            if tags:
                degraded_record["synthetic"] = True
                degraded_record["degradation_tags"] = tags
            degraded.append(degraded_record)
            summary["degraded_tracks"] += 1
        return degraded, summary

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load a JSON file, returning ``{}`` when missing or unreadable."""
        if not path.is_file():
            return {}
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write a JSON file, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


def main(argv: list[str] | None = None) -> None:
    """Generate degraded synthetic copies of the real perception outputs."""
    parser = argparse.ArgumentParser(
        description="Generate degraded synthetic perception outputs for "
        "benchmarking downstream identity fusion."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic degradation (e.g. --seed 42).",
    )
    parser.add_argument(
        "--input-dir",
        default=config.SYNTHETIC_INPUT_DIR,
        help="Directory containing the real observations.json and trajectories.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.SYNTHETIC_OUTPUT_DIR,
        help="Directory for degraded synthetic outputs.",
    )
    args = parser.parse_args(argv)

    if not config.SYNTHETIC_ENABLE:
        print(
            "Synthetic degradation is disabled. Set SYNTHETIC_ENABLE=True in "
            "perception/config.py to generate synthetic benchmark data."
        )
        return

    generator = SyntheticDegradationGenerator(seed=args.seed)
    summary = generator.generate(args.input_dir, args.output_dir)
    print(f"Synthetic outputs written to {args.output_dir}")
    print(f"Degraded frames: {summary['degraded_frames']}")
    print(f"Degraded tracks: {summary['degraded_tracks']}")
    print(f"Missing plates: {summary['missing_plate_count']}")
    print(f"OCR failures: {summary['ocr_failure_count']}")
    print(f"Missing embeddings: {summary['missing_embedding_count']}")
    print(f"Occluded vehicles: {summary['occluded_vehicle_count']}")
    print(f"Fragmented tracks: {summary['fragmented_track_count']}")
    print(f"Dropped frames: {summary['dropped_frame_count']}")
    print(f"Camera outage intervals: {summary['camera_outage_intervals']}")


if __name__ == "__main__":
    main()