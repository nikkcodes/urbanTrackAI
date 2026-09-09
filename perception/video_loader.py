"""OpenCV-based video input and metadata access."""

from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoLoader:
    """Read video frames and expose basic video metadata."""

    def __init__(self) -> None:
        """Initialize an unopened video loader."""
        self._capture: cv2.VideoCapture | None = None

    def open(self, path: str | PathLike[str]) -> None:
        """Open a video file.

        Args:
            path: Path to the video file.

        Raises:
            OSError: If the path does not exist or OpenCV cannot open it.
        """
        video_path = Path(path)
        if not video_path.is_file():
            raise OSError(f"Video file does not exist: {video_path}")

        self.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Unable to open video file: {video_path}")

        self._capture = capture

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame from the opened video.

        Returns:
            A success flag and the decoded BGR frame. At end of stream, the
            success flag is false and the frame is ``None``.

        Raises:
            RuntimeError: If no video is currently open.
        """
        capture = self._require_open()
        success, frame = capture.read()
        return success, frame if success else None

    def get_fps(self) -> float:
        """Return the opened video's frames-per-second value.

        Raises:
            RuntimeError: If no video is currently open.
        """
        return float(self._require_open().get(cv2.CAP_PROP_FPS))

    def get_frame_size(self) -> tuple[int, int]:
        """Return the opened video's frame size as ``(width, height)``.

        Raises:
            RuntimeError: If no video is currently open.
        """
        capture = self._require_open()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self) -> None:
        """Release the currently opened video, if any."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _require_open(self) -> Any:
        """Return the capture object or raise when no video is open."""
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("No video is open. Call open(path) first.")
        return self._capture
