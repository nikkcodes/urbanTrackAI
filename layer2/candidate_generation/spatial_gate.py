"""
UrbanTrack AI — Layer 2 Candidate Generation: Spatial Gate & Speed Feasibility.

Calculates camera-to-camera geographic distance and evaluates physically plausible
vehicle-speed bounds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from layer2.candidate_generation.topology_gate import TopologyEvidence

# Standard WGS-84 Earth radius in meters
EARTH_RADIUS_METERS = 6371000.0

# Documented maximum vehicle speed: 45.0 m/s (~162 km/h / 100.7 mph)
DEFAULT_MAX_SPEED_MPS = 45.0


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle Haversine distance in meters between two coordinates.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


@dataclass
class SpatialFeasibilityResult:
    """Spatial distance and implied speed feasibility assessment."""
    camera_distance_m: float
    implied_speed_mps: Optional[float]
    speed_feasible: bool
    rejection_reason_code: Optional[str] = None
    rejection_reason_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_distance_m": self.camera_distance_m,
            "implied_speed_mps": self.implied_speed_mps,
            "speed_feasible": self.speed_feasible,
        }


class SpatialGate:
    """
    Calculates distance between cameras and assesses physical vehicle speed feasibility.
    """

    def __init__(
        self,
        locations_path: Union[str, Path] = "UrbanTrack_Member1_Handoff 2/data/config/camera_locations.json",
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        adjacent_distance_m: float = 50.0,
    ) -> None:
        self.locations_path = Path(locations_path)
        self.max_speed_mps = float(max_speed_mps)
        self.adjacent_distance_m = float(adjacent_distance_m)
        self._locations: Dict[str, Tuple[float, float]] = {}
        self._load_locations()

    def _load_locations(self) -> None:
        if not self.locations_path.is_file():
            raise FileNotFoundError(f"Camera locations file not found at {self.locations_path}")

        with open(self.locations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cam_id, info in data.items():
            loc = info.get("location", {})
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat is not None and lon is not None:
                self._locations[cam_id] = (float(lat), float(lon))

    def get_camera_coordinates(self, camera_id: str) -> Optional[Tuple[float, float]]:
        return self._locations.get(camera_id)

    def calculate_distance_m(
        self,
        origin_cam: str,
        destination_cam: str,
        topology_evidence: Optional[TopologyEvidence] = None,
    ) -> float:
        """
        Calculates distance between two cameras in meters.
        Prefers directed graph edge distance when available; otherwise calculates Haversine distance.
        """
        if (
            topology_evidence is not None
            and topology_evidence.has_directed_edge
            and topology_evidence.edge_distance_m is not None
        ):
            return float(topology_evidence.edge_distance_m)

        if origin_cam not in self._locations:
            raise KeyError(f"Missing coordinates for origin camera: {origin_cam}")
        if destination_cam not in self._locations:
            raise KeyError(f"Missing coordinates for destination camera: {destination_cam}")

        lat1, lon1 = self._locations[origin_cam]
        lat2, lon2 = self._locations[destination_cam]
        return haversine_distance_m(lat1, lon1, lat2, lon2)

    def evaluate_speed_feasibility(
        self,
        camera_distance_m: float,
        delta_t_seconds: float,
        is_overlapping: bool = False,
    ) -> SpatialFeasibilityResult:
        """
        Evaluates speed feasibility given camera distance, elapsed travel time, and overlap status.

        Rules:
        - is_overlapping or delta_t <= 0: Admitted as concurrent observation. No inter-camera transit time.
        - delta_t > 0:
          - camera_distance_m <= adjacent_distance_m: Admitted as FOV boundary handoff / intersection transit.
          - camera_distance_m > adjacent_distance_m: Computes implied speed = distance / delta_t.
            Rejects if speed > max_speed_mps.
        """
        dist_rounded = round(camera_distance_m, 2)

        if is_overlapping or delta_t_seconds <= 0.0:
            return SpatialFeasibilityResult(
                camera_distance_m=dist_rounded,
                implied_speed_mps=None,
                speed_feasible=True,
                rejection_reason_code=None,
                rejection_reason_detail=None,
            )

        implied_speed = camera_distance_m / delta_t_seconds
        speed_rounded = round(implied_speed, 2)

        # Geometry-aware FOV boundary handoff:
        # Adjacent intersection cameras share visual borders; sub-second transitions traverse
        # inter-cone gaps (< 5m), not pole-to-pole distance.
        if camera_distance_m <= self.adjacent_distance_m:
            return SpatialFeasibilityResult(
                camera_distance_m=dist_rounded,
                implied_speed_mps=speed_rounded,
                speed_feasible=True,
                rejection_reason_code=None,
                rejection_reason_detail=None,
            )

        if implied_speed > self.max_speed_mps:
            return SpatialFeasibilityResult(
                camera_distance_m=dist_rounded,
                implied_speed_mps=speed_rounded,
                speed_feasible=False,
                rejection_reason_code="REJECT_EXCESSIVE_SPEED",
                rejection_reason_detail=(
                    f"Implied speed {speed_rounded:.2f} m/s ({speed_rounded * 3.6:.1f} km/h) "
                    f"exceeds maximum speed threshold {self.max_speed_mps:.2f} m/s"
                ),
            )

        return SpatialFeasibilityResult(
            camera_distance_m=dist_rounded,
            implied_speed_mps=speed_rounded,
            speed_feasible=True,
            rejection_reason_code=None,
            rejection_reason_detail=None,
        )
