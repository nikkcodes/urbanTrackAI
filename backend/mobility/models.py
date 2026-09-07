"""Data models for UrbanTrackAI road network representation.

Provides typed and validated Node and RoadSegment structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.mobility.config import (
    MAX_CAPACITY_VPH,
    MAX_DISTANCE_KM,
    MAX_SPEED_KMPH,
    MIN_CAPACITY_VPH,
    MIN_DISTANCE_KM,
    MIN_SPEED_KMPH,
)


@dataclass(slots=True)
class Node:
    """Represents an intersection, junction, or sensor location in the road network."""

    node_id: str
    name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate node attributes."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string.")

        self.node_id = self.node_id.strip()

        if self.lat is not None:
            if not isinstance(self.lat, (int, float)) or math.isnan(self.lat) or math.isinf(self.lat):
                raise ValueError(f"lat must be a valid finite float, got: {self.lat}")
            if not (-90.0 <= float(self.lat) <= 90.0):
                raise ValueError(f"lat must be between -90.0 and 90.0, got: {self.lat}")
            self.lat = float(self.lat)

        if self.lon is not None:
            if not isinstance(self.lon, (int, float)) or math.isnan(self.lon) or math.isinf(self.lon):
                raise ValueError(f"lon must be a valid finite float, got: {self.lon}")
            if not (-180.0 <= float(self.lon) <= 180.0):
                raise ValueError(f"lon must be between -180.0 and 180.0, got: {self.lon}")
            self.lon = float(self.lon)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        data: Dict[str, Any] = {
            "node_id": self.node_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "metadata": dict(self.metadata),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Node:
        """Deserialize node from dictionary."""
        return cls(
            node_id=data["node_id"],
            name=data.get("name"),
            lat=data.get("lat"),
            lon=data.get("lon"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class RoadSegment:
    """Represents a directed road segment connecting two junctions."""

    road_id: str
    from_node: str
    to_node: str
    distance_km: float
    speed_limit_kmph: float
    capacity_vph: float
    free_flow_time_min: Optional[float] = None
    is_closed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate road segment attributes and calculate free-flow travel time."""
        if not isinstance(self.road_id, str) or not self.road_id.strip():
            raise ValueError("road_id must be a non-empty string.")
        self.road_id = self.road_id.strip()

        if not isinstance(self.from_node, str) or not self.from_node.strip():
            raise ValueError("from_node must be a non-empty string.")
        self.from_node = self.from_node.strip()

        if not isinstance(self.to_node, str) or not self.to_node.strip():
            raise ValueError("to_node must be a non-empty string.")
        self.to_node = self.to_node.strip()

        if self.from_node == self.to_node:
            raise ValueError(f"Self-loop detected: from_node and to_node cannot be identical ('{self.from_node}').")

        # Validate distance
        if not isinstance(self.distance_km, (int, float)) or math.isnan(self.distance_km) or math.isinf(self.distance_km):
            raise ValueError(f"distance_km must be a valid finite number, got: {self.distance_km}")
        self.distance_km = float(self.distance_km)
        if self.distance_km <= 0:
            raise ValueError(f"distance_km must be strictly positive (> 0), got: {self.distance_km}")
        if self.distance_km > MAX_DISTANCE_KM:
            raise ValueError(f"distance_km exceeds maximum threshold ({MAX_DISTANCE_KM} km): {self.distance_km}")

        # Validate speed limit
        if not isinstance(self.speed_limit_kmph, (int, float)) or math.isnan(self.speed_limit_kmph) or math.isinf(self.speed_limit_kmph):
            raise ValueError(f"speed_limit_kmph must be a valid finite number, got: {self.speed_limit_kmph}")
        self.speed_limit_kmph = float(self.speed_limit_kmph)
        if self.speed_limit_kmph <= 0:
            raise ValueError(f"speed_limit_kmph must be strictly positive (> 0), got: {self.speed_limit_kmph}")
        if self.speed_limit_kmph > MAX_SPEED_KMPH:
            raise ValueError(f"speed_limit_kmph exceeds maximum threshold ({MAX_SPEED_KMPH} km/h): {self.speed_limit_kmph}")

        # Validate capacity
        if not isinstance(self.capacity_vph, (int, float)) or math.isnan(self.capacity_vph) or math.isinf(self.capacity_vph):
            raise ValueError(f"capacity_vph must be a valid finite number, got: {self.capacity_vph}")
        self.capacity_vph = float(self.capacity_vph)
        if self.capacity_vph <= 0:
            raise ValueError(f"capacity_vph must be strictly positive (> 0), got: {self.capacity_vph}")
        if self.capacity_vph > MAX_CAPACITY_VPH:
            raise ValueError(f"capacity_vph exceeds maximum threshold ({MAX_CAPACITY_VPH} vph): {self.capacity_vph}")

        # Calculate or validate free-flow travel time (minutes)
        expected_free_flow_min = (self.distance_km / self.speed_limit_kmph) * 60.0
        if self.free_flow_time_min is None:
            self.free_flow_time_min = round(expected_free_flow_min, 4)
        else:
            if not isinstance(self.free_flow_time_min, (int, float)) or math.isnan(self.free_flow_time_min) or math.isinf(self.free_flow_time_min):
                raise ValueError(f"free_flow_time_min must be a valid finite number, got: {self.free_flow_time_min}")
            self.free_flow_time_min = float(self.free_flow_time_min)
            if self.free_flow_time_min <= 0:
                raise ValueError(f"free_flow_time_min must be strictly positive (> 0), got: {self.free_flow_time_min}")

        if not isinstance(self.is_closed, bool):
            raise ValueError(f"is_closed must be a boolean, got: {type(self.is_closed)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize road segment to dictionary."""
        return {
            "road_id": self.road_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "distance_km": self.distance_km,
            "speed_limit_kmph": self.speed_limit_kmph,
            "capacity_vph": self.capacity_vph,
            "free_flow_time_min": self.free_flow_time_min,
            "is_closed": self.is_closed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadSegment:
        """Deserialize road segment from dictionary."""
        return cls(
            road_id=data["road_id"],
            from_node=data["from_node"],
            to_node=data["to_node"],
            distance_km=float(data["distance_km"]),
            speed_limit_kmph=float(data["speed_limit_kmph"]),
            capacity_vph=float(data["capacity_vph"]),
            free_flow_time_min=float(data["free_flow_time_min"]) if "free_flow_time_min" in data and data["free_flow_time_min"] is not None else None,
            is_closed=bool(data.get("is_closed", False)),
            metadata=dict(data.get("metadata", {})),
        )
