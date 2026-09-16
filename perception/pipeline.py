"""Video perception pipeline orchestration."""

import json
import re
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .config import (
    CAMERA_ID,
    CAMERA_METADATA,
    CAMERA_METADATA_PATH,
    CONFIDENCE_THRESHOLD,
    DETECTOR_MODEL,
    EXPORT_TRACK_EMBEDDINGS,
    FPS_SOURCE,
    HUD_ACCENT_COLOR,
    HUD_BACKGROUND_COLOR,
    HUD_FONT_SCALE_LIMITS,
    HUD_OPACITY,
    HUD_PADDING,
    HUD_RADIUS,
    HUD_TEXT_COLOR,
    LABEL_FONT_SCALE_LIMITS,
    LABEL_BACKGROUND_COLOR,
    MIN_LINE_THICKNESS,
    MIN_REID_CROP_SIZE,
    OCR_CACHE_FRAMES,
    OCR_CONF_THRESHOLD,
    OCR_MODEL,
    PLATE_BOX_COLOR,
    PLATE_CONF_THRESHOLD,
    PLATE_LABEL_COLOR,
    REFERENCE_FRAME_HEIGHT,
    REFERENCE_FRAME_WIDTH,
    REID_EMBEDDING_DIM,
    REID_MODEL_NAME,
    REID_MODEL_WEIGHTS,
    SYSTEM_TITLE,
    SUPPORTED_VIDEO_EXTENSIONS,
    TRAIL_GAP_PIXELS,
    INACTIVE_TRACK_MEMORY,
    TRAIL_LENGTH,
    TRAIL_MAX_THICKNESS,
    TRAIL_MIN_THICKNESS,
    TRACKER_MODEL,
    VEHICLE_COLORS,
)
from .vehicle_detector import VehicleDetector
from .video_loader import VideoLoader
from .plate_detector import PlateDetector
from .plate_ocr import read_plate
from .camera_calibration import CameraCalibration
from .camera_metrics import CameraMetrics
from .reid_extractor import ReIDExtractor


def _average_metric(frames: list[dict[str, object]], key: str) -> float:
    """Average a numeric metric across processed frames."""
    values = [float(frame[key]) for frame in frames]
    return sum(values) / len(values) if values else 0.0


class PerceptionPipeline:
    """Read video, detect vehicles, annotate frames, and write output."""

    def __init__(self) -> None:
        """Initialize the video loader, vehicle detector, and Re-ID extractor."""
        self._video_loader = VideoLoader()
        self._vehicle_detector = VehicleDetector(CONFIDENCE_THRESHOLD)
        self._plate_detector = PlateDetector()
        calibration_metadata = (
            CAMERA_METADATA.get(CAMERA_ID, {}) if CAMERA_ID else {}
        )
        self._camera_calibration = CameraCalibration(calibration_metadata)
        self._reid = ReIDExtractor() if EXPORT_TRACK_EMBEDDINGS else None
        self._track_embeddings: dict[int, tuple[list[float] | None, float | None]] = {}
        self._embedding_failures: int = 0

    def process(
        self,
        input_path: str | PathLike[str],
        output_path: str | PathLike[str] | None = None,
        camera_id: str | None = None,
    ) -> int:
        """Process a video and save its vehicle detections.

        Args:
            input_path: Path to the source video.
            output_path: Path for the processed video. If omitted, the output
                is written beside other outputs using the input filename stem.
            camera_id: Camera identifier that must exist in the camera
                metadata file. Defaults to ``CAMERA_ID`` (CAM_001).

        Raises:
            OSError: If the input or output video cannot be opened.
            ValueError: If ``camera_id`` is not a known camera in
                ``data/config/camera_metadata.json``.

        Returns:
            The number of processed frames.
        """
        camera_id = self._resolve_camera_id(camera_id)
        input_file = Path(input_path)
        output_dir = self._camera_output_dir(camera_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            Path(output_path)
            if output_path is not None
            else output_dir / f"{input_file.stem}_tracked{input_file.suffix}"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        observations_file = output_dir / "observations.json"
        observations_file.parent.mkdir(parents=True, exist_ok=True)
        trajectories_file = output_dir / "trajectories.json"
        camera_metrics_file = output_dir / "camera_metrics.json"
        perception_summary_file = output_dir / "perception_summary.json"
        writer: cv2.VideoWriter | None = None
        frame_count = 0
        frame_observations: list[dict[str, object]] = []
        camera_metric_frames: list[dict[str, object]] = []
        trajectory_records: dict[int, dict[str, object]] = {}
        previous_centroids: dict[int, tuple[int, int]] = {}
        configured_camera_metadata = (
            CAMERA_METADATA.get(camera_id, {}) if camera_id else {}
        )
        total_plate_detections = 0
        total_ocr_successes = 0
        total_ocr_failures = 0
        total_ocr_attempts = 0
        ocr_cache: dict[int, dict[str, str | float | int]] = {}
        ocr_last_attempt: dict[int, int] = {}
        track_history: dict[int, list[tuple[int, int]]] = {}
        inactive_track_age: dict[int, int] = {}
        track_colors: dict[int, tuple[int, int, int]] = {}
        self._track_embeddings = {}
        self._embedding_failures = 0

        try:
            self._video_loader.open(input_path)
            fps = self._video_loader.get_fps()
            frame_size = self._video_loader.get_frame_size()
            if fps <= 0:
                raise OSError(f"Unable to determine FPS for video: {input_path}")

            print(
                f"Camera calibrated: "
                f"{self._camera_calibration.camera_calibrated}"
            )
            if self._camera_calibration.homography_valid:
                print("Homography matrix loaded.")

            writer = cv2.VideoWriter(
                str(output_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise OSError(f"Unable to create output video: {output_file}")

            while True:
                success, frame = self._video_loader.read_frame()
                if not success or frame is None:
                    break

                detections = self._vehicle_detector.detect(frame)
                frame_metrics = CameraMetrics.compute(frame, detections)
                camera_metric_frames.append(
                    {
                        "frame_number": frame_count,
                        "timestamp": self._format_timestamp(frame_count / fps),
                        **frame_metrics,
                    }
                )
                plate_count = self._attach_plate_detections(frame, detections)
                ocr_success, ocr_failed, _ocr_debug = self._attach_plate_ocr(
                    frame,
                    detections,
                    frame_count,
                    ocr_cache,
                    ocr_last_attempt,
                )
                total_plate_detections += plate_count
                total_ocr_successes += ocr_success
                total_ocr_failures += ocr_failed
                active_track_ids = set()
                for detection in detections:
                    track_id = int(detection["track_id"])
                    x1, y1, x2, y2 = cast(list[int], detection["bbox"])
                    center = ((x1 + x2) // 2, y2)
                    active_track_ids.add(track_id)
                    inactive_track_age[track_id] = 0
                    track_colors[track_id] = VEHICLE_COLORS.get(
                        str(detection["vehicle_type"]), LABEL_BACKGROUND_COLOR
                    )
                    points = track_history.setdefault(track_id, [])
                    if not points or points[-1] != center:
                        points.append(center)
                    if len(points) > TRAIL_LENGTH:
                        del points[:-TRAIL_LENGTH]
                vehicles: list[dict[str, object]] = []
                for track_id in set(track_history) - active_track_ids:
                    inactive_track_age[track_id] = inactive_track_age.get(track_id, 0) + 1
                    if inactive_track_age[track_id] > INACTIVE_TRACK_MEMORY:
                        del track_history[track_id]
                        inactive_track_age.pop(track_id, None)
                        track_colors.pop(track_id, None)
                        previous_centroids.pop(track_id, None)
                for detection in detections:
                    vehicle_bbox = [
                        int(coordinate)
                        for coordinate in cast(list[int], detection["bbox"])
                    ]
                    plate_bbox_value = detection["plate_bbox"]
                    plate_bbox = (
                        [
                            int(coordinate)
                            for coordinate in cast(list[int], plate_bbox_value)
                        ]
                        if plate_bbox_value is not None
                        else None
                    )
                    x1, y1, x2, y2 = vehicle_bbox
                    plate_confidence = (
                        float(detection["plate_confidence"])
                        if plate_bbox is not None
                        else None
                    )
                    ocr_text = str(detection["plate_text"])
                    ocr_confidence = float(detection["ocr_confidence"])
                    ocr_succeeded = (
                        ocr_text != "UNKNOWN"
                        and ocr_confidence >= OCR_CONF_THRESHOLD
                    )
                    cleaned_plate_number = re.sub(
                        r"[^A-Z0-9]",
                        "",
                        ocr_text.upper(),
                    )
                    if (
                        not ocr_succeeded
                        or cleaned_plate_number == "UNKNOWN"
                        or len(cleaned_plate_number) < 6
                    ):
                        cleaned_plate_number = None
                    centroid = [(x1 + x2) // 2, y2]
                    track_id = int(detection["track_id"])

                    if EXPORT_TRACK_EMBEDDINGS and self._reid is not None:
                        if track_id not in self._track_embeddings:
                            frame_h, frame_w = frame.shape[:2]
                            crop_x1 = max(0, min(frame_w, x1))
                            crop_y1 = max(0, min(frame_h, y1))
                            crop_x2 = max(crop_x1, min(frame_w, x2))
                            crop_y2 = max(crop_y1, min(frame_h, y2))
                            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            emb, quality = self._reid.extract(vehicle_crop)
                            self._track_embeddings[track_id] = (emb, quality)
                            if emb is None:
                                self._embedding_failures += 1

                    current_centroid = (centroid[0], centroid[1])
                    previous_centroid = previous_centroids.get(track_id)
                    if previous_centroid is None:
                        velocity_vector = [0, 0]
                        velocity_px = 0.0
                        direction = "stationary"
                    else:
                        delta_x = current_centroid[0] - previous_centroid[0]
                        delta_y = current_centroid[1] - previous_centroid[1]
                        velocity_vector = [delta_x, delta_y]
                        velocity_px = round(
                            float(np.hypot(delta_x, delta_y)),
                            2,
                        )
                        if velocity_px < 2.0:
                            direction = "stationary"
                        elif abs(delta_y) > abs(delta_x):
                            direction = "southbound" if delta_y > 0 else "northbound"
                        else:
                            direction = "eastbound" if delta_x > 0 else "westbound"
                    previous_centroids[track_id] = current_centroid
                    trajectory_record = trajectory_records.setdefault(
                        track_id,
                        {
                            "track_id": track_id,
                            "vehicle_type": str(detection["vehicle_type"]),
                            "start_frame": frame_count,
                            "end_frame": frame_count,
                            "trajectory": [],
                            "velocity_sum": 0.0,
                            "velocity_count": 0,
                        },
                    )
                    trajectory_points = cast(
                        list[list[int]], trajectory_record["trajectory"]
                    )
                    if not trajectory_points or trajectory_points[-1] != centroid:
                        trajectory_points.append(centroid)
                        if len(trajectory_points) > TRAIL_LENGTH:
                            del trajectory_points[:-TRAIL_LENGTH]
                    trajectory_record["end_frame"] = frame_count
                    trajectory_record["velocity_sum"] = float(
                        trajectory_record["velocity_sum"]
                    ) + velocity_px
                    trajectory_record["velocity_count"] = int(
                        trajectory_record["velocity_count"]
                    ) + 1
                    ground_plane_position = (
                        self._camera_calibration.transform_to_ground(
                            centroid
                        )
                        if self._camera_calibration.camera_calibrated
                        else None
                    )
                    vehicles.append(
                        {
                            "track_id": track_id,
                            "embedding_id": track_id,
                            "reid_model": f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}",
                            "embedding_dim": REID_EMBEDDING_DIM,
                            "vehicle_type": str(detection["vehicle_type"]),
                            "confidence": float(detection["confidence"]),
                            "bbox": vehicle_bbox,
                            "centroid": centroid,
                            "velocity_px": velocity_px,
                            "velocity_vector": velocity_vector,
                            "direction": direction,
                            "plate_bbox": plate_bbox,
                            "plate_confidence": plate_confidence,
                            "plate_number": (
                                cleaned_plate_number
                            ),
                            "plate_text_confidence": (
                                ocr_confidence if ocr_succeeded else None
                            ),
                        }
                    )
                frame_observations.append(
                    {
                        "frame_number": frame_count,
                        "timestamp": self._format_timestamp(frame_count / fps),
                        "fps": fps,
                        "vehicle_count": len(vehicles),
                        "vehicles": vehicles,
                        "provenance": {
                            "source_video": input_file.name,
                            "camera_id": camera_id,
                            "frame_number": frame_count,
                            "detector_model": DETECTOR_MODEL,
                            "tracker_model": TRACKER_MODEL,
                            "ocr_model": OCR_MODEL,
                            "reid_model": f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}",
                            "processing_device": self._vehicle_detector.device,
                            "fps_source": FPS_SOURCE,
                        },
                    }
                )
                self._draw_info_overlay(
                    frame,
                    camera_id,
                    frame_count,
                    frame_count / fps,
                    fps,
                    detections,
                    plate_count,
                    ocr_success,
                    ocr_failed,
                    float(frame_metrics["reliability"]),
                )
                self._draw_detections(
                    frame,
                    detections,
                    track_history,
                    track_colors,
                    inactive_track_age,
                )
                writer.write(frame)
                frame_count += 1
        finally:
            self._video_loader.release()
            if writer is not None:
                writer.release()

        with observations_file.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "camera_id": camera_id,
                    "video_name": input_file.name,
                    "frames_processed": frame_count,
                    "frames": frame_observations,
                },
                file,
                indent=4,
            )

        trajectory_exports: list[dict[str, object]] = []
        for record in trajectory_records.values():
            start_frame = int(record["start_frame"])
            end_frame = int(record["end_frame"])
            points = cast(list[list[int]], record["trajectory"])
            velocity_count = int(record["velocity_count"])
            average_velocity = (
                float(record["velocity_sum"]) / velocity_count
                if velocity_count
                else 0.0
            )
            track_id = int(record["track_id"])
            appearance_emb, emb_quality = self._track_embeddings.get(
                track_id, (None, None)
            )
            trajectory_exports.append(
                {
                    "track_id": track_id,
                    "vehicle_type": str(record["vehicle_type"]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": end_frame - start_frame + 1,
                    "trajectory": points,
                    "trajectory_length": len(points),
                    "average_velocity_px": round(average_velocity, 2),
                    "appearance_embedding": appearance_emb,
                    "embedding_quality": emb_quality,
                    "embedding_dim": REID_EMBEDDING_DIM,
                    "reid_model": f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}",
                }
            )
        with trajectories_file.open("w", encoding="utf-8") as file:
            json.dump(trajectory_exports, file, indent=4)

        camera_metrics_export = [
            {
                **metric_frame,
                "vehicle_density": f"{float(metric_frame['vehicle_density']):.6f}",
            }
            for metric_frame in camera_metric_frames
        ]
        with camera_metrics_file.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "camera_id": camera_id,
                    "video_name": input_file.name,
                    "frame_width": frame_size[0],
                    "frame_height": frame_size[1],
                    "frames_processed": frame_count,
                    "frames": camera_metrics_export,
                },
                file,
                indent=4,
            )

        unique_track_ids = {
            int(vehicle["track_id"])
            for frame in frame_observations
            for vehicle in cast(list[dict[str, object]], frame["vehicles"])
        }
        vehicle_counts: dict[str, int] = {}
        total_vehicle_observations = 0
        frames_with_plates = 0
        frames_with_ocr_success = 0
        frame_track_counts: list[int] = []
        speeds: list[float] = []
        for frame in frame_observations:
            vehicles = cast(list[dict[str, object]], frame["vehicles"])
            frame_track_counts.append(len(vehicles))
            frame_has_plate = False
            frame_has_ocr_success = False
            for vehicle in vehicles:
                total_vehicle_observations += 1
                vehicle_type = str(vehicle["vehicle_type"])
                vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
                plate_bbox = vehicle.get("plate_bbox")
                if plate_bbox is not None:
                    frame_has_plate = True
                plate_number = vehicle.get("plate_number")
                if plate_number is not None:
                    frame_has_ocr_success = True
                speeds.append(float(vehicle["velocity_px"]))
            frames_with_plates += int(frame_has_plate)
            frames_with_ocr_success += int(frame_has_ocr_success)

        for vehicle_type in ("car", "bus", "truck", "motorcycle", "auto"):
            vehicle_counts.setdefault(vehicle_type, 0)
        unique_vehicle_counts: dict[str, int] = {}
        for record in trajectory_records.values():
            vehicle_type = str(record["vehicle_type"])
            unique_vehicle_counts[vehicle_type] = (
                unique_vehicle_counts.get(vehicle_type, 0) + 1
            )
        for vehicle_type in ("car", "bus", "truck", "motorcycle", "auto"):
            unique_vehicle_counts.setdefault(vehicle_type, 0)
        longest_track = max(
            trajectory_exports,
            key=lambda record: int(record["trajectory_length"]),
            default=None,
        )
        reliability_values = [
            float(frame["reliability"]) for frame in camera_metric_frames
        ]
        total_ocr_attempts = total_plate_detections
        total_ocr_failures = max(0, total_ocr_attempts - total_ocr_successes)
        success_rate = (
            (total_ocr_successes / total_ocr_attempts) * 100
            if total_ocr_attempts
            else 0.0
        )
        embeddings_generated = sum(
            1 for emb, _ in self._track_embeddings.values() if emb is not None
        )
        embedding_failures = sum(
            1 for emb, _ in self._track_embeddings.values() if emb is None
        )
        perception_summary_file = output_dir / "perception_summary.json"
        with perception_summary_file.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "camera_id": camera_id,
                    "video_name": input_file.name,
                    "frame_width": frame_size[0],
                    "frame_height": frame_size[1],
                    "frames_processed": frame_count,
                    "duration_seconds": frame_count / fps if fps else 0.0,
                    "average_fps": fps,
                    "processing_device": self._vehicle_detector.device,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "unique_tracks": len(unique_track_ids),
                    "active_tracks_peak": max(frame_track_counts, default=0),
                    "average_tracks_per_frame": (
                        sum(frame_track_counts) / len(frame_track_counts)
                        if frame_track_counts
                        else 0.0
                    ),
                    "vehicle_counts": vehicle_counts,
                    "unique_vehicle_counts": unique_vehicle_counts,
                    "plates_detected": total_plate_detections,
                    "ocr_attempts": total_ocr_attempts,
                    "ocr_success_count": total_ocr_successes,
                    "ocr_failure_count": total_ocr_failures,
                    "ocr_success_rate": round(success_rate, 2),
                    "reid_model": f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}",
                    "embedding_dimension": REID_EMBEDDING_DIM,
                    "embeddings_generated": embeddings_generated,
                    "embedding_generation_failures": embedding_failures,
                    "average_camera_reliability": (
                        sum(reliability_values) / len(reliability_values)
                        if reliability_values
                        else 0.0
                    ),
                    "minimum_camera_reliability": min(reliability_values, default=0.0),
                    "maximum_camera_reliability": max(reliability_values, default=0.0),
                    "average_brightness": _average_metric(camera_metric_frames, "brightness"),
                    "average_blur_score": _average_metric(camera_metric_frames, "blur_score"),
                    "average_occlusion_ratio": _average_metric(
                        camera_metric_frames, "occlusion_ratio"
                    ),
                    "average_vehicle_speed_px": (
                        sum(speeds) / len(speeds) if speeds else 0.0
                    ),
                    "longest_trajectory_track_id": (
                        int(longest_track["track_id"]) if longest_track else None
                    ),
                    "longest_trajectory_points": (
                        int(longest_track["trajectory_length"]) if longest_track else 0
                    ),
                    "average_trajectory_length": (
                        sum(int(record["trajectory_length"]) for record in trajectory_exports)
                        / len(trajectory_exports)
                        if trajectory_exports
                        else 0.0
                    ),
                    "frames_with_no_vehicles": sum(count == 0 for count in frame_track_counts),
                    "frames_with_plates": frames_with_plates,
                    "frames_with_ocr_success": frames_with_ocr_success,
                    "total_vehicle_observations": total_vehicle_observations,
                },
                file,
                indent=4,
            )

        print(f"Observations exported: {frame_count} frames")
        vehicles_exported = sum(
            int(frame["vehicle_count"]) for frame in frame_observations
        )
        print(f"Vehicles exported: {vehicles_exported}")
        print(f"JSON saved: {observations_file}")
        print(f"Plate detections: {total_plate_detections}")
        print(f"OCR successes: {total_ocr_successes}")
        print(f"OCR failures: {total_ocr_failures}")
        print(f"Motion vectors exported for {vehicles_exported} vehicle observations.")
        average_trajectory_length = (
            round(
                sum(len(cast(list[list[int]], record["trajectory"]))
                    for record in trajectory_records.values())
                / len(trajectory_records)
            )
            if trajectory_records
            else 0
        )
        print(f"Trajectories exported: {len(trajectory_exports)} tracks")
        print(f"Average trajectory length: {average_trajectory_length} points")
        print(f"JSON saved: {trajectories_file}")
        reid_model_str = f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}"
        print(f"Re-ID model: {reid_model_str}")
        print(f"Embedding dimension: {REID_EMBEDDING_DIM}")
        print(f"Embeddings generated: {embeddings_generated}")
        print(f"Embeddings failed: {embedding_failures}")
        average_reliability = (
            sum(float(item["reliability"]) for item in camera_metric_frames)
            / len(camera_metric_frames)
            if camera_metric_frames
            else 0.0
        )
        print(f"Camera metrics exported: {len(camera_metric_frames)} frames")
        print(f"Average camera reliability: {average_reliability:.2f}")
        print(f"JSON saved: {camera_metrics_file}")
        print("Perception summary exported.")
        print(f"Unique tracks: {len(unique_track_ids)}")
        print(
            "Average camera reliability: "
            f"{sum(reliability_values) / len(reliability_values):.2f}"
            if reliability_values
            else "Average camera reliability: 0.00"
        )
        print(f"OCR success rate: {success_rate:.2f}")
        print(f"Summary saved: {perception_summary_file}")

        return frame_count

    @staticmethod
    def _format_timestamp(timestamp_seconds: float) -> str:
        """Format video time as HH:MM:SS.mmm."""
        total_milliseconds = round(timestamp_seconds * 1000)
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _attach_plate_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, str | float | int | list[int]]],
    ) -> int:
        """Detect the strongest plate in each vehicle crop and attach its box."""
        plate_count = 0
        frame_height, frame_width = frame.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = cast(list[int], detection["bbox"])
            crop_x1 = max(0, min(frame_width, x1))
            crop_y1 = max(0, min(frame_height, y1))
            crop_x2 = max(crop_x1, min(frame_width, x2))
            crop_y2 = max(crop_y1, min(frame_height, y2))
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            detection["plate_bbox"] = None
            detection["plate_confidence"] = None
            if crop.size == 0:
                continue

            plates = [
                plate
                for plate in self._plate_detector.detect(crop)
                if float(plate["confidence"]) >= PLATE_CONF_THRESHOLD
            ]
            if not plates:
                continue

            plate = max(plates, key=lambda item: float(item["confidence"]))
            plate_x1, plate_y1, plate_x2, plate_y2 = cast(
                list[int], plate["bbox"]
            )
            detection["plate_bbox"] = [
                crop_x1 + int(plate_x1),
                crop_y1 + int(plate_y1),
                crop_x1 + int(plate_x2),
                crop_y1 + int(plate_y2),
            ]
            detection["plate_confidence"] = float(plate["confidence"])
            plate_count += 1
        return plate_count

    def _attach_plate_ocr(
        self,
        frame: np.ndarray,
        detections: list[dict[str, str | float | int | list[int]]],
        frame_number: int,
        ocr_cache: dict[int, dict[str, str | float | int]],
        ocr_last_attempt: dict[int, int],
    ) -> tuple[int, int, dict[str, str | float | int | bool]]:
        """Read eligible plate crops, reusing stable per-track OCR results."""
        success_count = 0
        failed_count = 0
        debug: dict[str, str | float | int | bool] = {
            "track_id": 0,
            "plate_text": "UNKNOWN",
            "confidence": 0.0,
            "cache_updated": False,
        }
        frame_height, frame_width = frame.shape[:2]
        eligible_types = {"car", "bus", "truck", "auto"}
        for detection in detections:
            track_id = int(detection["track_id"])
            debug["track_id"] = track_id
            detection["plate_text"] = "UNKNOWN"
            detection["ocr_confidence"] = 0.0
            plate_bbox = detection.get("plate_bbox")
            vehicle_type = str(detection["vehicle_type"]).lower()
            if not isinstance(plate_bbox, list):
                continue
            if vehicle_type not in eligible_types:
                failed_count += 1
                continue

            cached = ocr_cache.get(track_id)
            cache_is_fresh = (
                cached is not None
                and frame_number - int(cached["last_seen_frame"]) < OCR_CACHE_FRAMES
            )
            last_attempt = ocr_last_attempt.get(track_id)
            retry_is_due = (
                last_attempt is None
                or frame_number - last_attempt >= OCR_CACHE_FRAMES
            )
            if cache_is_fresh or not retry_is_due:
                ocr_result = cached
            else:
                plate_x1, plate_y1, plate_x2, plate_y2 = cast(list[int], plate_bbox)
                crop_x1 = max(0, min(frame_width, plate_x1))
                crop_y1 = max(0, min(frame_height, plate_y1))
                crop_x2 = max(crop_x1, min(frame_width, plate_x2))
                crop_y2 = max(crop_y1, min(frame_height, plate_y2))
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                result = read_plate(crop)
                ocr_last_attempt[track_id] = frame_number
                candidate_text = str(result["plate_number"])
                candidate_confidence = float(result["confidence"])
                cache_updated = False
                if (
                    candidate_text != "UNKNOWN"
                    and candidate_confidence >= OCR_CONF_THRESHOLD
                    and (
                        cached is None
                        or candidate_confidence > float(cached["ocr_confidence"])
                    )
                ):
                    ocr_cache[track_id] = {
                        "plate_text": candidate_text,
                        "ocr_confidence": candidate_confidence,
                        "last_seen_frame": frame_number,
                    }
                    cache_updated = True
                if cache_updated or not bool(debug["cache_updated"]):
                    debug = {
                        "track_id": track_id,
                        "plate_text": candidate_text,
                        "confidence": candidate_confidence,
                        "cache_updated": cache_updated,
                    }
                ocr_result = ocr_cache.get(track_id, cached)

            if ocr_result is None:
                ocr_result = {
                    "plate_text": "UNKNOWN",
                    "ocr_confidence": 0.0,
                }

            detection["plate_text"] = str(ocr_result["plate_text"])
            detection["ocr_confidence"] = float(ocr_result["ocr_confidence"])
            if debug["plate_text"] == "UNKNOWN" and detection["plate_text"] != "UNKNOWN":
                debug["track_id"] = track_id
                debug["plate_text"] = detection["plate_text"]
                debug["confidence"] = detection["ocr_confidence"]
            if (
                detection["plate_text"] != "UNKNOWN"
                and float(ocr_result["ocr_confidence"]) >= OCR_CONF_THRESHOLD
            ):
                success_count += 1
            else:
                failed_count += 1
        return success_count, failed_count, debug

    @staticmethod
    def _find_input_video(input_directory: Path) -> Path | None:
        """Return the only video or the newest supported input video."""
        if not input_directory.is_dir():
            return None

        videos = [
            path
            for path in input_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]
        return max(
            videos,
            key=lambda path: (path.stat().st_mtime, path.name.lower()),
            default=None,
        )

    @staticmethod
    def _load_camera_metadata() -> dict[str, dict[str, object]]:
        """Load the camera metadata file, returning ``{camera_id: {...}}``.

        Returns an empty dict when the file is missing or unreadable.
        """
        metadata_path = Path(CAMERA_METADATA_PATH)
        if not metadata_path.is_file():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, ValueError):
            return {}
        cameras = data.get("cameras") if isinstance(data, dict) else None
        if not isinstance(cameras, list):
            return {}
        result: dict[str, dict[str, object]] = {}
        for entry in cameras:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("camera_id")
            if isinstance(cid, str) and cid:
                result[cid] = entry
        return result

    @classmethod
    def _resolve_camera_id(cls, camera_id: str | None) -> str:
        """Validate and resolve the requested camera id.

        Args:
            camera_id: Requested camera id, or ``None`` for the default.

        Returns:
            The resolved camera id.

        Raises:
            ValueError: If ``camera_id`` is not present in the camera metadata
                file. Metadata is never silently created.
        """
        resolved = camera_id if camera_id else CAMERA_ID
        known = cls._load_camera_metadata()
        if resolved not in known:
            raise ValueError(
                f"Unknown camera_id '{resolved}'. Known cameras: "
                f"{', '.join(sorted(known)) if known else 'none (no metadata file)'}"
            )
        return resolved

    @staticmethod
    def _find_camera_video(camera_id: str, input_directory: Path) -> Path | None:
        """Return the newest supported video for a camera, if any.

        Looks first inside ``<input_directory>/<camera_id>/`` and falls back
        to a flat ``<input_directory>`` for backward compatibility.
        """
        camera_directory = input_directory / camera_id
        if camera_directory.is_dir():
            video = PerceptionPipeline._find_input_video(camera_directory)
            if video is not None:
                return video
        return PerceptionPipeline._find_input_video(input_directory)

    @staticmethod
    def _camera_output_dir(camera_id: str, base_dir: str | Path = "data/output") -> Path:
        """Return the camera-specific output directory, creating it if needed."""
        output_dir = Path(base_dir) / camera_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _draw_info_overlay(
        frame: np.ndarray,
        camera_id: str | None,
        frame_number: int,
        timestamp: float,
        fps: float,
        detections: list[dict[str, str | float | int | list[int]]],
        plate_count: int,
        ocr_success: int,
        ocr_failed: int,
        camera_health: float,
    ) -> None:
        """Draw current camera and tracking statistics in a translucent panel."""
        frame_height, frame_width = frame.shape[:2]
        frame_scale = min(
            frame_width / REFERENCE_FRAME_WIDTH,
            frame_height / REFERENCE_FRAME_HEIGHT,
        )
        line_thickness = max(MIN_LINE_THICKNESS, round(frame_scale))
        font_scale = min(
            HUD_FONT_SCALE_LIMITS[1],
            max(HUD_FONT_SCALE_LIMITS[0], frame_scale),
        )
        padding = max(6, round(HUD_PADDING * frame_scale))
        line_height = max(round(24 * frame_scale), padding * 2)
        title = SYSTEM_TITLE
        title_size, title_baseline = cv2.getTextSize(
            title,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            line_thickness,
        )

        counts = {"car": 0, "auto": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        for detection in detections:
            vehicle_type = str(detection["vehicle_type"]).lower()
            if vehicle_type in counts:
                counts[vehicle_type] += 1

        lines = [
            f"Camera ID: {camera_id}",
            f"Frame: {frame_number}",
            f"Timestamp: {int(timestamp) // 3600:02d}:{int(timestamp) % 3600 // 60:02d}:{int(timestamp) % 60:02d}",
            f"FPS: {fps:.2f}",
            f"Active tracked vehicles: {len(detections)}",
            f"Cars: {counts['car']}",
            f"Auto: {counts['auto']}",
            f"Buses: {counts['bus']}",
            f"Trucks: {counts['truck']}",
            f"Motorcycles: {counts['motorcycle']}",
            f"Plates Detected: {plate_count}",
            f"OCR Success: {ocr_success}",
            f"OCR Failed: {ocr_failed}",
            f"Camera Health: {camera_health:.2f}",
        ]
        text_sizes = [
            cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                line_thickness,
            )[0]
            for line in lines
        ]
        panel_width = min(
            frame_width,
            max(text_width for text_width, _ in text_sizes) + padding * 2,
        )
        panel_height = min(
            frame_height,
            title_size[1] + title_baseline + line_height * len(lines) + padding * 2,
        )

        overlay = frame.copy()
        PerceptionPipeline._draw_rounded_rectangle(
            overlay,
            (0, 0),
            (panel_width - 1, panel_height - 1),
            HUD_BACKGROUND_COLOR,
            max(1, round(HUD_RADIUS * frame_scale)),
        )
        title_y = padding + title_size[1]
        cv2.putText(
            overlay,
            title,
            (padding, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            HUD_TEXT_COLOR,
            line_thickness,
            cv2.LINE_AA,
        )
        accent_y = title_y + padding // 2
        cv2.line(
            overlay,
            (padding, accent_y),
            (panel_width - padding, accent_y),
            HUD_ACCENT_COLOR,
            line_thickness,
        )
        health_color = (
            (0, 200, 0)
            if camera_health >= 0.80
            else (0, 220, 220)
            if camera_health >= 0.60
            else (0, 0, 255)
        )
        for line_number, line in enumerate(lines):
            text_y = title_y + padding + (line_number + 1) * line_height
            cv2.putText(
                overlay,
                line,
                (padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                health_color if line.startswith("Camera Health:") else HUD_TEXT_COLOR,
                line_thickness,
                cv2.LINE_AA,
            )

        frame[:panel_height, :panel_width] = cv2.addWeighted(
            overlay[:panel_height, :panel_width],
            HUD_OPACITY,
            frame[:panel_height, :panel_width],
            1 - HUD_OPACITY,
            0,
        )

    @staticmethod
    def _draw_detections(
        frame: np.ndarray,
        detections: list[dict[str, str | float | int | list[int]]],
        track_history: dict[int, list[tuple[int, int]]],
        track_colors: dict[int, tuple[int, int, int]],
        inactive_track_age: dict[int, int],
    ) -> None:
        """Draw resolution-aware boxes and compact readable labels on a frame."""
        frame_height, frame_width = frame.shape[:2]
        font_scale = min(
            LABEL_FONT_SCALE_LIMITS[1],
            max(LABEL_FONT_SCALE_LIMITS[0], frame_height / 1800),
        )
        line_thickness = max(1, frame_width // 1200)
        label_padding = 4

        active_track_ids = {int(detection["track_id"]) for detection in detections}
        for track_id, points in track_history.items():
            vehicle_color = track_colors.get(track_id, LABEL_BACKGROUND_COLOR)
            historical_points = points[:-1] if track_id in active_track_ids else points
            PerceptionPipeline._draw_trail(
                frame,
                historical_points,
                vehicle_color,
                points[-1] if track_id in active_track_ids and points else None,
                inactive_track_age.get(track_id, 0),
            )

        for detection in detections:
            x1, y1, x2, y2 = cast(list[int], detection["bbox"])
            track_id = int(detection["track_id"])
            vehicle_type = cast(str, detection["vehicle_type"])
            confidence = float(detection["confidence"])
            vehicle_color = VEHICLE_COLORS.get(vehicle_type, LABEL_BACKGROUND_COLOR)
            plate_bbox = detection.get("plate_bbox")
            if isinstance(plate_bbox, list):
                plate_x1, plate_y1, plate_x2, plate_y2 = cast(list[int], plate_bbox)
                plate_width = max(1, plate_x2 - plate_x1)
                plate_text = str(detection.get("plate_text", "UNKNOWN"))
                plate_font_scale = max(0.6, plate_width / 80.0)
                plate_line_thickness = 3
                (plate_text_width, plate_text_height), plate_baseline = (
                    cv2.getTextSize(
                        plate_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        plate_font_scale,
                        plate_line_thickness,
                    )
                )
                cv2.rectangle(
                    frame,
                    (plate_x1, plate_y1),
                    (plate_x2, plate_y2),
                    PLATE_BOX_COLOR,
                    plate_line_thickness,
                    cv2.LINE_AA,
                )
                plate_label_padding = 4
                plate_label_width = plate_text_width + plate_label_padding * 2
                plate_label_height = (
                    plate_text_height
                    + plate_baseline
                    + plate_label_padding * 2
                )
                plate_label_x = min(
                    max(0, (plate_x1 + plate_x2 - plate_label_width) // 2),
                    max(0, frame_width - plate_label_width),
                )
                plate_label_top = plate_y2 + 2
                if plate_label_top + plate_label_height > frame_height:
                    plate_label_top = max(0, plate_y1 - plate_label_height - 2)
                plate_label_bottom = min(
                    frame_height - 1,
                    plate_label_top + plate_label_height,
                )
                PerceptionPipeline._draw_rounded_rectangle(
                    frame,
                    (plate_label_x, plate_label_top),
                    (
                        min(frame_width - 1, plate_label_x + plate_label_width),
                        plate_label_bottom,
                    ),
                    (20, 20, 20),
                    max(2, round(plate_label_height * 0.2)),
                )
                cv2.putText(
                    frame,
                    plate_text,
                    (
                        plate_label_x + plate_label_padding,
                        plate_label_top + plate_label_padding + plate_text_height,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    plate_font_scale,
                    PLATE_LABEL_COLOR,
                    plate_line_thickness,
                    cv2.LINE_AA,
                )
            label = f"ID {track_id} • {vehicle_type.upper()} • {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                line_thickness,
            )
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                vehicle_color,
                line_thickness,
            )
            label_width = text_width + label_padding * 2
            label_height = text_height + baseline + label_padding * 2
            label_x = min(max(0, x1), max(0, frame_width - label_width))
            label_top = y1 - label_height
            if label_top >= 0:
                label_y = y1 - label_padding - baseline
            else:
                label_top = min(y1 + label_padding, frame_height - label_height)
                label_y = label_top + label_padding + text_height
            label_bottom = min(frame_height - 1, label_top + label_height)
            PerceptionPipeline._draw_rounded_rectangle(
                frame,
                (label_x, label_top),
                (min(frame_width - 1, label_x + label_width), label_bottom),
                (20, 20, 20),
                max(2, round(4 * font_scale)),
            )
            cv2.putText(
                frame,
                label,
                (label_x + label_padding, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_trail(
        frame: np.ndarray,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        current_point: tuple[int, int] | None,
        inactive_age: int,
    ) -> None:
        """Draw a fading historical trajectory, leaving space at its head."""
        if len(points) < 2:
            return

        trail_points = points[:]
        if current_point is not None:
            gap = TRAIL_GAP_PIXELS
            previous_point = trail_points[-1]
            delta_x = previous_point[0] - current_point[0]
            delta_y = previous_point[1] - current_point[1]
            distance = float(np.hypot(delta_x, delta_y))
            if distance > 0:
                trail_points[-1] = (
                    round(previous_point[0] + gap * delta_x / distance),
                    round(previous_point[1] + gap * delta_y / distance),
                )

        segment_count = len(trail_points) - 1
        for segment_index, (start, end) in enumerate(
            zip(trail_points, trail_points[1:])
        ):
            age_progress = (segment_index + 1) / segment_count
            thickness = round(
                TRAIL_MIN_THICKNESS
                + (TRAIL_MAX_THICKNESS - TRAIL_MIN_THICKNESS) * age_progress
            )
            brightness = 0.62 + 0.38 * age_progress
            if inactive_age:
                brightness *= max(
                    0.08,
                    1.0 - inactive_age / INACTIVE_TRACK_MEMORY,
                )
            faded_color = tuple(round(channel * brightness) for channel in color)
            cv2.line(frame, start, end, faded_color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_rounded_rectangle(
        frame: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        """Draw a filled rounded rectangle using OpenCV primitives."""
        left, top = top_left
        right, bottom = bottom_right
        radius = min(radius, (right - left) // 2, (bottom - top) // 2)
        cv2.rectangle(
            frame,
            (left + radius, top),
            (right - radius, bottom),
            color,
            cv2.FILLED,
        )
        cv2.rectangle(
            frame,
            (left, top + radius),
            (right, bottom - radius),
            color,
            cv2.FILLED,
        )
        for center in (
            (left + radius, top + radius),
            (right - radius, top + radius),
            (left + radius, bottom - radius),
            (right - radius, bottom - radius),
        ):
            cv2.circle(frame, center, radius, color, cv2.FILLED)


def main(argv: list[str] | None = None) -> None:
    """Run the UrbanTrack perception pipeline on the default input video.

    Accepts an optional ``--camera-id`` argument selecting one of the
    prototype cameras defined in ``data/config/camera_metadata.json``
    (defaults to ``CAMERA_001``). Videos may live flat in
    ``data/input/`` or under ``data/input/<camera_id>/``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the UrbanTrack perception pipeline for one camera."
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Camera identifier (e.g. CAM_001). Defaults to CAM_001.",
    )
    args = parser.parse_args(argv)

    input_directory = Path("data/input")
    try:
        camera_id = PerceptionPipeline._resolve_camera_id(args.camera_id)
    except ValueError as error:
        print(f"Error: {error}")
        return

    input_path = PerceptionPipeline._find_camera_video(camera_id, input_directory)

    print("Starting UrbanTrack Perception Pipeline...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Camera ID: {camera_id}")

    if input_path is None:
        print(
            f"Error: no supported video found for camera {camera_id} in "
            f"{input_directory} or {input_directory / camera_id}."
        )
        return

    output_path = PerceptionPipeline._camera_output_dir(camera_id) / (
        f"{input_path.stem}_tracked{input_path.suffix}"
    )
    print(f"Selected input filename: {input_path.name}")
    print(f"Output video path: {output_path}")

    pipeline = PerceptionPipeline()
    frame_count = pipeline.process(input_path, output_path, camera_id=camera_id)
    print(f"Total processed frames: {frame_count}")


if __name__ == "__main__":
    main()
