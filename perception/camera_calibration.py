"""Camera calibration and image-to-ground coordinate transformation."""

from typing import Any

import cv2
import numpy as np


class CameraCalibration:
    """Compute and apply a homography for one camera."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        """Initialize calibration from configured image and ground points."""
        self.camera_id = metadata.get("camera_id") if metadata else None
        self.camera_calibrated = bool(
            metadata.get("camera_calibrated", False) if metadata else False
        )
        self._homography: np.ndarray | None = None
        image_points = metadata.get("image_points", []) if metadata else []
        ground_points = metadata.get("ground_points", []) if metadata else []
        if self.camera_calibrated and len(image_points) >= 4 and len(ground_points) >= 4:
            source = np.asarray(image_points, dtype=np.float32)
            target = np.asarray(ground_points, dtype=np.float32)
            if source.shape == target.shape and source.shape[1:] == (2,):
                self._homography, _ = cv2.findHomography(source, target)
        self.camera_calibrated = self._homography is not None

    @property
    def homography_valid(self) -> bool:
        """Return whether a usable homography is loaded."""
        return self._homography is not None

    def transform_to_ground(
        self,
        point: tuple[int, int] | list[int],
    ) -> list[float] | None:
        """Transform an image point to ground coordinates when calibrated."""
        if self._homography is None:
            return None
        source = np.asarray([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, self._homography)[0, 0]
        return [float(transformed[0]), float(transformed[1])]
