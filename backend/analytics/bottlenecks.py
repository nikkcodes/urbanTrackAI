"""Deterministic bottleneck ranking from Phase 3 traffic metrics."""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from backend.analytics.models import Bottleneck
from backend.traffic.models import CongestionLevel, TrafficMetric


class BottleneckDetector:
    """Rank measurable road pressure without making predictive or AI claims.

    The score is:

        w_u * utilization_ratio
        + w_c * congestion_score
        + w_d * (delay_minutes / free_flow_time_minutes)

    The three non-negative weights are explicit configuration. Each component
    increases the score when its corresponding measurable pressure increases.
    """

    def __init__(
        self,
        utilization_weight: float = 1.0,
        congestion_weight: float = 1.0,
        delay_weight: float = 1.0,
    ) -> None:
        self.utilization_weight = self._validate_weight(utilization_weight, "utilization_weight")
        self.congestion_weight = self._validate_weight(congestion_weight, "congestion_weight")
        self.delay_weight = self._validate_weight(delay_weight, "delay_weight")
        if self.utilization_weight + self.congestion_weight + self.delay_weight <= 0.0:
            raise ValueError("at least one bottleneck score weight must be positive")

    def detect(self, metrics: Iterable[TrafficMetric]) -> Tuple[Bottleneck, ...]:
        """Return all provided road metrics ranked by severity and score."""
        records = list(metrics)
        bottlenecks = [self._build_bottleneck(metric) for metric in records]
        severity_rank = {
            CongestionLevel.SEVERE.value: 3,
            CongestionLevel.HEAVY.value: 2,
            CongestionLevel.MODERATE.value: 1,
            CongestionLevel.FREE.value: 0,
        }
        bottlenecks.sort(key=lambda item: (-severity_rank.get(item.severity, -1), -item.bottleneck_score, item.road_id))
        return tuple(bottlenecks)

    def _build_bottleneck(self, metric: TrafficMetric) -> Bottleneck:
        if not isinstance(metric, TrafficMetric):
            raise TypeError(f"Expected TrafficMetric, got: {type(metric)}")
        if metric.hourly_flow is None or not math.isfinite(metric.hourly_flow) or metric.hourly_flow < 0.0:
            raise ValueError(f"TrafficMetric '{metric.road_id}' must provide a valid hourly_flow")
        delay_minutes = max(0.0, metric.estimated_travel_time_minutes - metric.free_flow_time_minutes)
        delay_ratio = delay_minutes / metric.free_flow_time_minutes
        score = (
            self.utilization_weight * metric.utilization_ratio
            + self.congestion_weight * metric.congestion_score
            + self.delay_weight * delay_ratio
        )
        signals: List[str] = []
        if metric.utilization_ratio >= 1.0:
            signals.append("utilization_at_or_above_capacity")
        if metric.congestion_score > 0.0:
            signals.append("bpr_delay_present")
        if delay_minutes > 0.0:
            signals.append("travel_time_above_free_flow")
        if not signals:
            signals.append("low_measured_pressure")

        severity = metric.congestion_level.value if isinstance(metric.congestion_level, CongestionLevel) else str(metric.congestion_level)
        return Bottleneck(
            road_id=metric.road_id,
            severity=severity,
            utilization_ratio=metric.utilization_ratio,
            congestion_score=metric.congestion_score,
            hourly_flow=metric.hourly_flow,
            capacity_vph=metric.capacity_vph,
            delay_minutes=delay_minutes,
            bottleneck_score=score,
            signals=tuple(signals),
        )

    @staticmethod
    def _validate_weight(value: float, name: str) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return float(value)
