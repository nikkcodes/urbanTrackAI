"""Video perception pipeline orchestration."""

import json
from os import PathLike
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .config import (
    CAMERA_ID,
    CONFIDENCE_THRESHOLD,
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
    OCR_CACHE_FRAMES,
    OCR_CONF_THRESHOLD,
    PLATE_BOX_COLOR,
    PLATE_CONF_THRESHOLD,
    PLATE_LABEL_COLOR,
    REFERENCE_FRAME_HEIGHT,
    REFERENCE_FRAME_WIDTH,
    SYSTEM_TITLE,
    SUPPORTED_VIDEO_EXTENSIONS,
    TRAIL_GAP_PIXELS,
    INACTIVE_TRACK_MEMORY,
    TRAIL_LENGTH,
    TRAIL_MAX_THICKNESS,
    TRAIL_MIN_THICKNESS,
    VEHICLE_COLORS,
)
from .vehicle_detector import VehicleDetector
from .video_loader import VideoLoader
from .plate_detector import PlateDetector
from .plate_ocr import read_plate


class PerceptionPipeline:
    """Read video, detect vehicles, annotate frames, and write output."""

    def __init__(self) -> None:
        """Initialize the video loader and vehicle detector."""
        self._video_loader = VideoLoader()
        self._vehicle_detector = VehicleDetector(CONFIDENCE_THRESHOLD)
        self._plate_detector = PlateDetector()

    def process(
        self,
        input_path: str | PathLike[str],
        output_path: str | PathLike[str] | None = None,
    ) -> int:
        """Process a video and save its vehicle detections.

        Args:
            input_path: Path to the source video.
            output_path: Path for the processed video. If omitted, the output
                is written beside other outputs using the input filename stem.

        Raises:
            OSError: If the input or output video cannot be opened.

        Returns:
            The number of processed frames.
        """
        input_file = Path(input_path)
        output_file = (
            Path(output_path)
            if output_path is not None
            else Path("data/output") / f"{input_file.stem}_tracked{input_file.suffix}"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        observations_file = Path("data/output/observations.json")
        observations_file.parent.mkdir(parents=True, exist_ok=True)
        writer: cv2.VideoWriter | None = None
        frame_count = 0
        observations: list[dict[str, str | float | int | list[int] | None]] = []
        camera_id = CAMERA_ID if CAMERA_ID else None
        total_plate_detections = 0
        total_ocr_successes = 0
        total_ocr_failures = 0
        ocr_cache: dict[int, dict[str, str | float | int]] = {}
        ocr_last_attempt: dict[int, int] = {}
        track_history: dict[int, list[tuple[int, int]]] = {}
        inactive_track_age: dict[int, int] = {}
        track_colors: dict[int, tuple[int, int, int]] = {}

        try:
            self._video_loader.open(input_path)
            fps = self._video_loader.get_fps()
            frame_size = self._video_loader.get_frame_size()
            if fps <= 0:
                raise OSError(f"Unable to determine FPS for video: {input_path}")

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
                    points.append(center)
                    if len(points) > TRAIL_LENGTH:
                        del points[:-TRAIL_LENGTH]
                for track_id in set(track_history) - active_track_ids:
                    inactive_track_age[track_id] = inactive_track_age.get(track_id, 0) + 1
                    if inactive_track_age[track_id] > INACTIVE_TRACK_MEMORY:
                        del track_history[track_id]
                        inactive_track_age.pop(track_id, None)
                        track_colors.pop(track_id, None)
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
                    observations.append(
                        {
                            "camera_id": camera_id,
                            "frame_number": frame_count,
                            "timestamp": frame_count / fps,
                            "track_id": int(detection["track_id"]),
                            "vehicle_type": str(detection["vehicle_type"]),
                            "detection_confidence": float(detection["confidence"]),
                            "bbox": vehicle_bbox,
                            "trajectory_point": [
                                (x1 + x2) // 2,
                                y2,
                            ],
                            "plate_bbox": plate_bbox,
                            "plate_confidence": plate_confidence,
                            "plate_text": (
                                ocr_text if ocr_succeeded else None
                            ),
                            "ocr_confidence": (
                                ocr_confidence if ocr_succeeded else None
                            ),
                            "appearance_embedding": None,
                            "latitude": None,
                            "longitude": None,
                            "camera_reliability": None,
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
            json.dump(observations, file, indent=4)

        print(f"Observations exported: {frame_count} frames")
        print(f"Vehicles exported: {len(observations)}")
        print(f"Plate detections: {total_plate_detections}")
        print(f"OCR successes: {total_ocr_successes}")
        print(f"OCR failures: {total_ocr_failures}")

        return frame_count

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
                candidate_text = str(result["text"])
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
        """Return the first supported input video in alphabetical order."""
        if not input_directory.is_dir():
            return None

        videos = sorted(
            (
                path
                for path in input_directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        return videos[0] if videos else None

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
        for line_number, line in enumerate(lines):
            text_y = title_y + padding + (line_number + 1) * line_height
            cv2.putText(
                overlay,
                line,
                (padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                HUD_TEXT_COLOR,
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
                plate_label_top = plate_y1 - plate_label_height - 2
                if plate_label_top < 0:
                    plate_label_top = min(
                        frame_height - plate_label_height,
                        plate_y2 + 2,
                    )
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


def main() -> None:
    """Run the UrbanTrack perception pipeline on the default input video."""
    input_directory = Path("data/input")
    input_path = PerceptionPipeline._find_input_video(input_directory)

    print("Starting UrbanTrack Perception Pipeline...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    if input_path is None:
        supported_formats = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        print(
            f"Error: no supported video found in {input_directory}. "
            f"Supported formats: {supported_formats}"
        )
        return

    output_path = Path("data/output") / f"{input_path.stem}_tracked{input_path.suffix}"
    print(f"Selected input filename: {input_path.name}")
    print(f"Output video path: {output_path}")

    pipeline = PerceptionPipeline()
    frame_count = pipeline.process(input_path, output_path)
    print(f"Total processed frames: {frame_count}")


if __name__ == "__main__":
    main()
