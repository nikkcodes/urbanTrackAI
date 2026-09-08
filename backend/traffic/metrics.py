"""Traffic intelligence metric calculations for UrbanTrackAI Phase 3.

Implements BPR-based congested travel time calculation, capacity utilization ratios,
normalized congestion scoring, discrete congestion leveling, and network-level summaries.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from typing import Dict, Iterable, List, Optional, Tuple

from backend.flow.models import FlowAggregationResult, RoadFlow
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.traffic.models import (
    BPRParameters,
    CongestionLevel,
    CongestionThresholds,
    NetworkTrafficSummary,
    TrafficMetric,
)


class TrafficMetricsCalculator:
    """Calculates road-level traffic performance metrics and network-wide summaries.

    Attributes:
        graph: Optional MobilityGraph reference to look up road physical attributes (capacity, free-flow time).
        bpr_params: BPR function parameters (alpha, beta). Defaults to alpha=0.15, beta=4.0.
        thresholds: Congestion thresholds for mapping utilization to CongestionLevel.
    """

    def __init__(
        self,
        graph: Optional[MobilityGraph] = None,
        bpr_params: Optional[BPRParameters] = None,
        thresholds: Optional[CongestionThresholds] = None,
        default_aggregation_duration_hours: float = 1.0,
    ) -> None:
        """Initialize TrafficMetricsCalculator.

        Args:
            graph: Optional MobilityGraph instance for looking up segment physical properties.
            bpr_params: Optional BPRParameters instance.
            thresholds: Optional CongestionThresholds instance.
            default_aggregation_duration_hours: Explicit duration used only when a RoadFlow
                has no time-window metadata. Defaults to 1 hour.
        """
        if graph is not None and not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph instance or None, got: {type(graph)}")

        self.graph: Optional[MobilityGraph] = graph
        self.bpr_params: BPRParameters = bpr_params if bpr_params is not None else BPRParameters()
        self.thresholds: CongestionThresholds = thresholds if thresholds is not None else CongestionThresholds()
        if (
            not isinstance(default_aggregation_duration_hours, (int, float))
            or not math.isfinite(default_aggregation_duration_hours)
            or default_aggregation_duration_hours <= 0.0
        ):
            raise ValueError("default_aggregation_duration_hours must be a finite positive number")
        self.default_aggregation_duration_hours = float(default_aggregation_duration_hours)

    @staticmethod
    def _parse_time_window_value(value: str, field_name: str) -> datetime | time:
        """Parse an ISO date-time or time-only window boundary."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty ISO time value")
        raw_value = value.strip()
        try:
            return datetime.fromisoformat(raw_value)
        except ValueError:
            try:
                return time.fromisoformat(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field_name} is not a valid ISO time value: {value!r}") from exc

    def _aggregation_duration_hours(self, road_flow: RoadFlow) -> float:
        """Return a validated window duration, or the explicit no-window fallback."""
        start = road_flow.time_window_start
        end = road_flow.time_window_end
        if start is None and end is None:
            return self.default_aggregation_duration_hours
        if start is None or end is None:
            raise ValueError("time_window_start and time_window_end must be provided together")

        parsed_start = self._parse_time_window_value(start, "time_window_start")
        parsed_end = self._parse_time_window_value(end, "time_window_end")
        if type(parsed_start) is not type(parsed_end):
            raise ValueError("time window boundaries must use the same date-time format")
        if isinstance(parsed_start, time):
            parsed_start = datetime.combine(date(2000, 1, 1), parsed_start)
            parsed_end = datetime.combine(date(2000, 1, 1), parsed_end)  # type: ignore[arg-type]
        try:
            duration_seconds = (parsed_end - parsed_start).total_seconds()  # type: ignore[operator]
        except TypeError as exc:
            raise ValueError("time window boundaries must have compatible timezone information") from exc
        if duration_seconds <= 0.0:
            raise ValueError("time window end must be after time window start")
        return duration_seconds / 3600.0

    def calculate_road_metric(
        self,
        road_flow: RoadFlow,
        road_segment: Optional[RoadSegment] = None,
    ) -> TrafficMetric:
        """Compute traffic metrics for a single RoadFlow record.

        Args:
            road_flow: RoadFlow instance from Phase 2 aggregation.
            road_segment: Optional RoadSegment providing physical attributes. If omitted,
                          looks up road_flow.road_id in self.graph.

        Returns:
            Computed TrafficMetric instance.

        Raises:
            ValueError: If capacity is missing/non-positive, or flow is invalid.
            KeyError: If road segment cannot be resolved in graph.
        """
        if not isinstance(road_flow, RoadFlow):
            raise TypeError(f"Expected RoadFlow instance, got: {type(road_flow)}")

        # Resolve road physical attributes
        segment = road_segment
        if segment is None:
            if self.graph is None:
                raise ValueError(
                    f"Cannot calculate metrics for road '{road_flow.road_id}': "
                    f"no road_segment provided and calculator was initialized without a MobilityGraph."
                )
            segment = self.graph.get_road(road_flow.road_id)
            if segment is None:
                raise KeyError(
                    f"Road segment '{road_flow.road_id}' not found in MobilityGraph."
                )

        capacity = segment.capacity_vph
        if not isinstance(capacity, (int, float)) or math.isnan(capacity) or math.isinf(capacity) or capacity <= 0:
            raise ValueError(
                f"Road '{segment.road_id}' has invalid capacity: {capacity}. "
                f"Capacity must be a strictly positive finite number (> 0)."
            )

        free_flow_time = segment.free_flow_time_min
        if free_flow_time is None or free_flow_time <= 0:
            # Derive if missing: (distance / speed) * 60
            if segment.speed_limit_kmph <= 0:
                raise ValueError(f"Road '{segment.road_id}' has non-positive speed limit: {segment.speed_limit_kmph}")
            free_flow_time = (segment.distance_km / segment.speed_limit_kmph) * 60.0

        flow = road_flow.expected_flow
        if not isinstance(flow, (int, float)) or math.isnan(flow) or math.isinf(flow) or flow < 0:
            raise ValueError(f"expected_flow must be a finite non-negative number, got: {flow}")

        duration_hours = self._aggregation_duration_hours(road_flow)
        hourly_flow = flow / duration_hours

        # 1. Capacity Utilization Ratio (hourly V / hourly C) - unclamped
        utilization_ratio = hourly_flow / capacity

        # 2. BPR Congested Travel Time: t = t0 * (1 + alpha * (V/C)^beta)
        alpha = self.bpr_params.alpha
        beta = self.bpr_params.beta
        delay_factor = alpha * (utilization_ratio ** beta)
        estimated_travel_time = free_flow_time * (1.0 + delay_factor)

        # 3. Congestion Score: fraction of travel time spent in congestion delay: (t - t0) / t
        # Equivalent to: delay_factor / (1.0 + delay_factor) = 1.0 - (t0 / t)
        if estimated_travel_time <= 0:
            congestion_score = 0.0
        else:
            delay_minutes = max(0.0, estimated_travel_time - free_flow_time)
            congestion_score = delay_minutes / estimated_travel_time

        # Ensure float bounds [0.0, 1.0)
        congestion_score = min(1.0, max(0.0, congestion_score))

        # 4. Congestion Level
        congestion_level = self.thresholds.classify(utilization_ratio)

        return TrafficMetric(
            road_id=road_flow.road_id,
            expected_flow=flow,
            hourly_flow=hourly_flow,
            capacity_vph=capacity,
            utilization_ratio=utilization_ratio,
            free_flow_time_minutes=free_flow_time,
            estimated_travel_time_minutes=estimated_travel_time,
            congestion_score=congestion_score,
            congestion_level=congestion_level,
            time_window_start=road_flow.time_window_start,
            time_window_end=road_flow.time_window_end,
            metadata={
                "distance_km": segment.distance_km,
                "speed_limit_kmph": segment.speed_limit_kmph,
                "bpr_alpha": alpha,
                "bpr_beta": beta,
                "aggregation_duration_hours": duration_hours,
            },
        )

    def calculate_metrics(self, flow_result: FlowAggregationResult) -> List[TrafficMetric]:
        """Compute traffic metrics for all road flows in a Phase 2 FlowAggregationResult.

        Roads are evaluated and returned in deterministic order sorted by road_id.

        Args:
            flow_result: FlowAggregationResult from Phase 2.

        Returns:
            List of TrafficMetric instances.
        """
        if not isinstance(flow_result, FlowAggregationResult):
            raise TypeError(f"Expected FlowAggregationResult, got: {type(flow_result)}")

        metrics: List[TrafficMetric] = []
        for rf in flow_result.all_flows(sorted_by_id=True):
            metric = self.calculate_road_metric(rf)
            metrics.append(metric)

        return metrics

    def summarize_network(
        self,
        metrics: Iterable[TrafficMetric],
        time_window_start: Optional[str] = None,
        time_window_end: Optional[str] = None,
    ) -> NetworkTrafficSummary:
        """Compute network-wide traffic summary statistics across a set of TrafficMetric records.

        Explicit mathematical definitions:
        - road_count: Total registered roads in the network (from MobilityGraph if available,
          otherwise count of evaluated roads).
        - active_roads_with_flow: Number of evaluated roads with expected_flow > 0.
        - evaluated_roads_count: Count of evaluated TrafficMetric items in this batch.
        - average_utilization: Simple arithmetic mean of utilization ratios across evaluated roads:
            sum(utilization_ratio) / evaluated_roads_count (0.0 if empty).
        - flow_weighted_utilization: Hourly-flow-weighted utilization ratio:
            sum(hourly_flow * utilization_ratio) / sum(hourly_flow)
            Defined as 0.0 when total hourly flow == 0.
        - average_travel_time_minutes: Simple arithmetic mean of estimated travel times across evaluated roads:
            sum(estimated_travel_time_minutes) / evaluated_roads_count (0.0 if empty).
        - flow_weighted_travel_time_minutes: Hourly-flow-weighted travel time:
            sum(hourly_flow * estimated_travel_time_minutes) / sum(hourly_flow)
            Defined as 0.0 when total hourly flow == 0.
        - congestion_level_counts: Total roads in each CongestionLevel category.

        Args:
            metrics: Iterable of TrafficMetric instances.
            time_window_start: Optional window start label.
            time_window_end: Optional window end label.

        Returns:
            NetworkTrafficSummary object.
        """
        metric_list = list(metrics)
        evaluated_count = len(metric_list)

        # Network road count preference: MobilityGraph total road count if available
        if self.graph is not None:
            total_network_roads = self.graph.road_count
        else:
            total_network_roads = evaluated_count

        total_flow = 0.0
        total_hourly_flow = 0.0
        active_flow_roads = 0
        sum_utilization = 0.0
        sum_flow_weighted_utilization = 0.0
        sum_travel_time = 0.0
        sum_flow_weighted_travel_time = 0.0

        level_counts: Dict[str, int] = {
            CongestionLevel.FREE.value: 0,
            CongestionLevel.MODERATE.value: 0,
            CongestionLevel.HEAVY.value: 0,
            CongestionLevel.SEVERE.value: 0,
        }

        # Inherit window labels if not explicitly passed
        win_start = time_window_start
        win_end = time_window_end
        if metric_list and win_start is None and win_end is None:
            starts = {m.time_window_start for m in metric_list if m.time_window_start}
            ends = {m.time_window_end for m in metric_list if m.time_window_end}
            if len(starts) == 1:
                win_start = next(iter(starts))
            if len(ends) == 1:
                win_end = next(iter(ends))

        for m in metric_list:
            flow = m.expected_flow
            hourly_flow = m.hourly_flow if m.hourly_flow is not None else flow
            total_flow += flow
            total_hourly_flow += hourly_flow
            if flow > 0:
                active_flow_roads += 1

            sum_utilization += m.utilization_ratio
            sum_flow_weighted_utilization += hourly_flow * m.utilization_ratio

            sum_travel_time += m.estimated_travel_time_minutes
            sum_flow_weighted_travel_time += hourly_flow * m.estimated_travel_time_minutes

            lvl = m.congestion_level.value if isinstance(m.congestion_level, CongestionLevel) else str(m.congestion_level)
            level_counts[lvl] = level_counts.get(lvl, 0) + 1

        # Safe division handling
        if evaluated_count > 0:
            avg_utilization = sum_utilization / evaluated_count
            avg_travel_time = sum_travel_time / evaluated_count
        else:
            avg_utilization = 0.0
            avg_travel_time = 0.0

        if total_hourly_flow > 0.0:
            flow_w_util = sum_flow_weighted_utilization / total_hourly_flow
            flow_w_time = sum_flow_weighted_travel_time / total_hourly_flow
        else:
            flow_w_util = 0.0
            flow_w_time = 0.0

        return NetworkTrafficSummary(
            total_expected_flow=total_flow,
            road_count=total_network_roads,
            active_roads_with_flow=active_flow_roads,
            evaluated_roads_count=evaluated_count,
            average_utilization=avg_utilization,
            flow_weighted_utilization=flow_w_util,
            average_travel_time_minutes=avg_travel_time,
            flow_weighted_travel_time_minutes=flow_w_time,
            congestion_level_counts=level_counts,
            time_window_start=win_start,
            time_window_end=win_end,
        )

    def calculate_windowed_metrics(
        self,
        windowed_flows: Dict[Tuple[Optional[str], Optional[str]], FlowAggregationResult],
    ) -> Dict[Tuple[Optional[str], Optional[str]], Tuple[List[TrafficMetric], NetworkTrafficSummary]]:
        """Compute metrics and summaries across multiple time-window partitions.

        Args:
            windowed_flows: Dict mapping (start, end) time window tuples to FlowAggregationResult.

        Returns:
            Dict mapping (start, end) tuples to (metrics_list, network_summary) tuples.
        """
        results: Dict[Tuple[Optional[str], Optional[str]], Tuple[List[TrafficMetric], NetworkTrafficSummary]] = {}

        for (start, end), flow_res in windowed_flows.items():
            metrics = self.calculate_metrics(flow_res)
            summary = self.summarize_network(metrics, time_window_start=start, time_window_end=end)
            results[(start, end)] = (metrics, summary)

        return results
