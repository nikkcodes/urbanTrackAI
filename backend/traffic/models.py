"""Data models for UrbanTrackAI Phase 3 traffic metrics and intelligence.

Provides typed structures for BPR parameters, congestion thresholds,
segment-level traffic metrics, and network-level summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class CongestionLevel(str, Enum):
    """Categorical classification of traffic congestion state."""

    FREE = "FREE"
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"
    SEVERE = "SEVERE"


@dataclass(slots=True)
class BPRParameters:
    """Configurable parameters for the Bureau of Public Roads (BPR) volume-delay function.

    Formula:
        t = t0 * (1 + alpha * (V / C) ** beta)

    Attributes:
        alpha: Coefficient scaling the delay term (standard default: 0.15).
        beta: Exponent governing the steepness of delay as V approaches and exceeds C (standard default: 4.0).
    """

    alpha: float = 0.15
    beta: float = 4.0

    def __post_init__(self) -> None:
        """Validate BPR parameters."""
        if not isinstance(self.alpha, (int, float)) or math.isnan(self.alpha) or math.isinf(self.alpha):
            raise ValueError(f"alpha must be a finite float, got: {self.alpha}")
        self.alpha = float(self.alpha)
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative (>= 0.0), got: {self.alpha}")

        if not isinstance(self.beta, (int, float)) or math.isnan(self.beta) or math.isinf(self.beta):
            raise ValueError(f"beta must be a finite float, got: {self.beta}")
        self.beta = float(self.beta)
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative (>= 0.0), got: {self.beta}")

    def to_dict(self) -> Dict[str, float]:
        """Serialize BPR parameters to dictionary."""
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BPRParameters:
        """Construct BPRParameters from dictionary."""
        return cls(
            alpha=float(data.get("alpha", 0.15)),
            beta=float(data.get("beta", 4.0)),
        )


@dataclass(slots=True)
class CongestionThresholds:
    """Thresholds for mapping volume-to-capacity (utilization) ratios to CongestionLevel.

    Default thresholds align with standard highway engineering Level of Service (LOS):
        - utilization < free_limit (0.70)       -> FREE
        - free_limit <= utilization < moderate  -> MODERATE
        - moderate <= utilization < heavy       -> HEAVY
        - utilization >= heavy (1.10)           -> SEVERE
    """

    free_limit: float = 0.70
    moderate_limit: float = 0.90
    heavy_limit: float = 1.10

    def __post_init__(self) -> None:
        """Validate thresholds ordering and ranges."""
        for name, val in [
            ("free_limit", self.free_limit),
            ("moderate_limit", self.moderate_limit),
            ("heavy_limit", self.heavy_limit),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be a finite float, got: {val}")

        self.free_limit = float(self.free_limit)
        self.moderate_limit = float(self.moderate_limit)
        self.heavy_limit = float(self.heavy_limit)

        if self.free_limit <= 0.0:
            raise ValueError(f"free_limit must be strictly positive (> 0.0), got: {self.free_limit}")

        if not (self.free_limit < self.moderate_limit < self.heavy_limit):
            raise ValueError(
                f"Thresholds must be strictly increasing: "
                f"free_limit ({self.free_limit}) < moderate_limit ({self.moderate_limit}) < heavy_limit ({self.heavy_limit})"
            )

    def classify(self, utilization: float) -> CongestionLevel:
        """Classify a utilization ratio into a discrete CongestionLevel."""
        if utilization < self.free_limit:
            return CongestionLevel.FREE
        elif utilization < self.moderate_limit:
            return CongestionLevel.MODERATE
        elif utilization < self.heavy_limit:
            return CongestionLevel.HEAVY
        else:
            return CongestionLevel.SEVERE

    def to_dict(self) -> Dict[str, float]:
        """Serialize thresholds to dictionary."""
        return {
            "free_limit": self.free_limit,
            "moderate_limit": self.moderate_limit,
            "heavy_limit": self.heavy_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CongestionThresholds:
        """Construct CongestionThresholds from dictionary."""
        return cls(
            free_limit=float(data.get("free_limit", 0.70)),
            moderate_limit=float(data.get("moderate_limit", 0.90)),
            heavy_limit=float(data.get("heavy_limit", 1.10)),
        )


@dataclass(slots=True)
class TrafficMetric:
    """Represents computed traffic performance metrics for a single road segment.

    Attributes:
        road_id: Unique identifier for the road segment.
        expected_flow: Expected vehicle-segment traversals in the aggregation window (from Phase 2).
        hourly_flow: Window-normalized expected road flow in vehicles per hour.
        capacity_vph: Design capacity in vehicles per hour.
        utilization_ratio: V / C ratio (unclamped).
        free_flow_time_minutes: Free-flow travel time (t0) from Phase 1.
        estimated_travel_time_minutes: Congested travel time from BPR function.
        congestion_score: Normalized delay fraction in [0.0, 1.0) derived as (t - t0) / t.
        congestion_level: Categorical state (FREE, MODERATE, HEAVY, SEVERE).
        time_window_start: Optional start of analysis window.
        time_window_end: Optional end of analysis window.
        metadata: Optional auxiliary information (road type, coordinates, etc.).
    """

    road_id: str
    expected_flow: float
    capacity_vph: float
    utilization_ratio: float
    free_flow_time_minutes: float
    estimated_travel_time_minutes: float
    congestion_score: float
    congestion_level: CongestionLevel
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hourly_flow: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate traffic metric fields."""
        if not isinstance(self.road_id, str) or not self.road_id.strip():
            raise ValueError("road_id must be a non-empty string.")
        self.road_id = self.road_id.strip()

        # Validate expected flow
        if not isinstance(self.expected_flow, (int, float)) or math.isnan(self.expected_flow) or math.isinf(self.expected_flow):
            raise ValueError(f"expected_flow must be a finite float, got: {self.expected_flow}")
        self.expected_flow = float(self.expected_flow)
        if self.expected_flow < 0.0:
            raise ValueError(f"expected_flow must be non-negative (>= 0.0), got: {self.expected_flow}")

        if self.hourly_flow is not None:
            if not isinstance(self.hourly_flow, (int, float)) or math.isnan(self.hourly_flow) or math.isinf(self.hourly_flow):
                raise ValueError(f"hourly_flow must be a finite float, got: {self.hourly_flow}")
            self.hourly_flow = float(self.hourly_flow)
            if self.hourly_flow < 0.0:
                raise ValueError(f"hourly_flow must be non-negative (>= 0.0), got: {self.hourly_flow}")

        # Validate capacity
        if not isinstance(self.capacity_vph, (int, float)) or math.isnan(self.capacity_vph) or math.isinf(self.capacity_vph):
            raise ValueError(f"capacity_vph must be a finite float, got: {self.capacity_vph}")
        self.capacity_vph = float(self.capacity_vph)
        if self.capacity_vph <= 0.0:
            raise ValueError(f"capacity_vph must be strictly positive (> 0.0), got: {self.capacity_vph}")

        # Validate utilization ratio
        if not isinstance(self.utilization_ratio, (int, float)) or math.isnan(self.utilization_ratio) or math.isinf(self.utilization_ratio):
            raise ValueError(f"utilization_ratio must be a finite float, got: {self.utilization_ratio}")
        self.utilization_ratio = float(self.utilization_ratio)
        if self.utilization_ratio < 0.0:
            raise ValueError(f"utilization_ratio must be non-negative (>= 0.0), got: {self.utilization_ratio}")

        # Validate free-flow time
        if not isinstance(self.free_flow_time_minutes, (int, float)) or math.isnan(self.free_flow_time_minutes) or math.isinf(self.free_flow_time_minutes):
            raise ValueError(f"free_flow_time_minutes must be a finite float, got: {self.free_flow_time_minutes}")
        self.free_flow_time_minutes = float(self.free_flow_time_minutes)
        if self.free_flow_time_minutes <= 0.0:
            raise ValueError(f"free_flow_time_minutes must be strictly positive (> 0.0), got: {self.free_flow_time_minutes}")

        # Validate estimated travel time
        if not isinstance(self.estimated_travel_time_minutes, (int, float)) or math.isnan(self.estimated_travel_time_minutes) or math.isinf(self.estimated_travel_time_minutes):
            raise ValueError(f"estimated_travel_time_minutes must be a finite float, got: {self.estimated_travel_time_minutes}")
        self.estimated_travel_time_minutes = float(self.estimated_travel_time_minutes)
        if self.estimated_travel_time_minutes < self.free_flow_time_minutes - 1e-6:
            raise ValueError(
                f"estimated_travel_time_minutes ({self.estimated_travel_time_minutes}) "
                f"cannot be less than free_flow_time_minutes ({self.free_flow_time_minutes})"
            )

        # Validate congestion score
        if not isinstance(self.congestion_score, (int, float)) or math.isnan(self.congestion_score) or math.isinf(self.congestion_score):
            raise ValueError(f"congestion_score must be a finite float, got: {self.congestion_score}")
        self.congestion_score = float(self.congestion_score)
        if not (0.0 <= self.congestion_score <= 1.0 + 1e-6):
            raise ValueError(f"congestion_score must be between 0.0 and 1.0, got: {self.congestion_score}")

        # Ensure congestion_level is CongestionLevel enum
        if isinstance(self.congestion_level, str):
            self.congestion_level = CongestionLevel(self.congestion_level)
        elif not isinstance(self.congestion_level, CongestionLevel):
            raise TypeError(f"congestion_level must be a CongestionLevel enum or str, got: {type(self.congestion_level)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize traffic metric to dictionary."""
        return {
            "road_id": self.road_id,
            "expected_flow": round(self.expected_flow, 4),
            "hourly_flow": round(self.hourly_flow, 4) if self.hourly_flow is not None else None,
            "capacity_vph": round(self.capacity_vph, 2),
            "utilization_ratio": round(self.utilization_ratio, 4),
            "free_flow_time_minutes": round(self.free_flow_time_minutes, 4),
            "estimated_travel_time_minutes": round(self.estimated_travel_time_minutes, 4),
            "congestion_score": round(self.congestion_score, 4),
            "congestion_level": self.congestion_level.value,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrafficMetric:
        """Construct TrafficMetric from dictionary."""
        return cls(
            road_id=data["road_id"],
            expected_flow=float(data["expected_flow"]),
            hourly_flow=float(data["hourly_flow"]) if data.get("hourly_flow") is not None else None,
            capacity_vph=float(data["capacity_vph"]),
            utilization_ratio=float(data["utilization_ratio"]),
            free_flow_time_minutes=float(data["free_flow_time_minutes"]),
            estimated_travel_time_minutes=float(data["estimated_travel_time_minutes"]),
            congestion_score=float(data["congestion_score"]),
            congestion_level=CongestionLevel(data["congestion_level"]),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class NetworkTrafficSummary:
    """Network-level aggregation of traffic metrics.

    Attributes:
        total_expected_flow: Sum of expected vehicle-segment traversals across evaluated roads.
        road_count: Total registered roads in the MobilityGraph (network size).
        active_roads_with_flow: Number of evaluated roads that have strictly positive flow (V > 0).
        evaluated_roads_count: Total number of road segments evaluated in this metric batch.
        average_utilization: Simple arithmetic average of utilization ratios across evaluated roads.
        flow_weighted_utilization: Flow-weighted average utilization ratio:
            sum(hourly_flow * utilization_ratio) / sum(hourly_flow)
            Defined as 0.0 when total hourly flow is 0.
        average_travel_time_minutes: Simple arithmetic average of estimated travel times across evaluated roads.
        flow_weighted_travel_time_minutes: Flow-weighted average travel time experienced by vehicles:
            sum(hourly_flow * estimated_travel_time) / sum(hourly_flow)
            Defined as 0.0 when total hourly flow is 0.
        congestion_level_counts: Number of evaluated roads grouped by CongestionLevel.
        time_window_start: Optional start of analysis window.
        time_window_end: Optional end of analysis window.
        metadata: Optional metadata.
    """

    total_expected_flow: float
    road_count: int
    active_roads_with_flow: int
    evaluated_roads_count: int
    average_utilization: float
    flow_weighted_utilization: float
    average_travel_time_minutes: float
    flow_weighted_travel_time_minutes: float
    congestion_level_counts: Dict[str, int]
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network summary to dictionary."""
        return {
            "total_expected_flow": round(self.total_expected_flow, 4),
            "road_count": self.road_count,
            "active_roads_with_flow": self.active_roads_with_flow,
            "evaluated_roads_count": self.evaluated_roads_count,
            "average_utilization": round(self.average_utilization, 4),
            "flow_weighted_utilization": round(self.flow_weighted_utilization, 4),
            "average_travel_time_minutes": round(self.average_travel_time_minutes, 4),
            "flow_weighted_travel_time_minutes": round(self.flow_weighted_travel_time_minutes, 4),
            "congestion_level_counts": dict(self.congestion_level_counts),
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }
