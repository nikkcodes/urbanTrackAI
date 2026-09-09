"""Video perception pipeline orchestration."""

import json
from os import PathLike
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from .config import (
    CONFIDENCE_THRESHOLD,
    HUD_ACCENT_COLOR,
    HUD_BACKGROUND_ALPHA,
    HUD_BACKGROUND_COLOR,
    HUD_TEXT_COLOR,
    LABEL_BACKGROUND_COLOR,
    MIN_FONT_SCALE,
    MIN_LINE_THICKNESS,
    REFERENCE_FRAME_HEIGHT,
    REFERENCE_FRAME_WIDTH,
    SUPPORTED_VIDEO_EXTENSIONS,
    VEHICLE_COLORS,
)
from .vehicle_detector import VehicleDetector
from .video_loader import VideoLoader


class PerceptionPipeline:
    """Read video, detect vehicles, annotate frames, and write output."""

    def __init__(self) -> None:
        """Initialize the video loader and vehicle detector."""
        self._video_loader = VideoLoader()
        self._vehicle_detector = VehicleDetector(CONFIDENCE_THRESHOLD)

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
            else Path("data/output") / f"{input_file.stem}_tracked.mp4"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        observations_file = Path("data/output/observations.json")
        observations_file.parent.mkdir(parents=True, exist_ok=True)
        writer: cv2.VideoWriter | None = None
        frame_count = 0
        observations: list[dict[str, str | float | int | list[int]]] = []
        camera_id = Path(input_path).stem

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
                for detection in detections:
                    observations.append(
                        {
                            "camera_id": camera_id,
                            "frame_id": frame_count,
                            "timestamp_seconds": frame_count / fps,
                            "track_id": int(detection["track_id"]),
                            "vehicle_type": str(detection["vehicle_type"]),
                            "confidence": float(detection["confidence"]),
                            "bbox": [
                                int(coordinate)
                                for coordinate in cast(
                                    list[int], detection["bbox"]
                                )
                            ],
                        }
                    )
                self._draw_info_overlay(
                    frame,
                    camera_id,
                    frame_count,
                    frame_count / fps,
                    fps,
                    detections,
                )
                self._draw_detections(frame, detections)
                writer.write(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
        finally:
            self._video_loader.release()
            if writer is not None:
                writer.release()

        with observations_file.open("w", encoding="utf-8") as file:
            json.dump(observations, file, indent=4)

        return frame_count

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
        camera_id: str,
        frame_number: int,
        timestamp: float,
        fps: float,
        detections: list[dict[str, str | float | int | list[int]]],
    ) -> None:
        """Draw current camera and tracking statistics in a translucent panel."""
        frame_height, frame_width = frame.shape[:2]
        frame_scale = min(
            frame_width / REFERENCE_FRAME_WIDTH,
            frame_height / REFERENCE_FRAME_HEIGHT,
        )
        line_thickness = max(MIN_LINE_THICKNESS, round(frame_scale))
        font_scale = max(MIN_FONT_SCALE, frame_scale)
        padding = max(6, round(12 * frame_scale))
        line_height = max(round(24 * frame_scale), padding * 2)
        title = "UrbanTrack AI"
        title_size, title_baseline = cv2.getTextSize(
            title,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            line_thickness,
        )

        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        for detection in detections:
            vehicle_type = str(detection["vehicle_type"]).lower()
            if vehicle_type in counts:
                counts[vehicle_type] += 1

        lines = [
            f"Camera ID: {camera_id}",
            f"Frame: {frame_number}",
            f"Timestamp: {int(timestamp) // 3600:02d}:{int(timestamp) % 3600 // 60:02d}:{int(timestamp) % 60:02d}",
            f"FPS: {fps:.2f}",
            f"Total active vehicles: {len(detections)}",
            f"Cars: {counts['car']}",
            f"Buses: {counts['bus']}",
            f"Trucks: {counts['truck']}",
            f"Motorcycles: {counts['motorcycle']}",
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
        cv2.rectangle(
            overlay,
            (0, 0),
            (panel_width - 1, panel_height - 1),
            HUD_BACKGROUND_COLOR,
            cv2.FILLED,
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
            HUD_BACKGROUND_ALPHA,
            frame[:panel_height, :panel_width],
            1 - HUD_BACKGROUND_ALPHA,
            0,
        )

    @staticmethod
    def _draw_detections(
        frame: np.ndarray,
        detections: list[dict[str, str | float | int | list[int]]],
    ) -> None:
        """Draw resolution-aware boxes and compact readable labels on a frame."""
        frame_height, frame_width = frame.shape[:2]
        font_scale = max(0.45, min(frame_height / 1800, 0.8))
        line_thickness = max(1, frame_width // 1200)
        label_padding = 4

        for detection in detections:
            x1, y1, x2, y2 = cast(list[int], detection["bbox"])
            track_id = int(detection["track_id"])
            vehicle_type = cast(str, detection["vehicle_type"])
            confidence = float(detection["confidence"])
            vehicle_color = VEHICLE_COLORS.get(vehicle_type, LABEL_BACKGROUND_COLOR)
            label = f"ID {track_id} | {vehicle_type.title()} | {confidence:.2f}"
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
                4,
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

    output_path = Path("data/output") / f"{input_path.stem}_tracked.mp4"
    print(f"Selected input filename: {input_path.name}")
    print(f"Output video path: {output_path}")

    pipeline = PerceptionPipeline()
    frame_count = pipeline.process(input_path, output_path)
    print(f"Total processed frames: {frame_count}")


if __name__ == "__main__":
    main()
