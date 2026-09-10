"""
Trajectory Anomaly Detector for UrbanTrack AI (Day 7).

Evaluates individual NormalizedTrajectory records against configured baseline models,
road network physical limits, and Day 3 feasibility constraints.

CRITICAL SEMANTIC PRINCIPLES:
1. INVALID DATA ≠ PHYSICAL INCONSISTENCY ≠ BEHAVIORAL ANOMALY ≠ INVESTIGATION PRIORITY.
2. Malformed input (e.g. invalid timestamps, temporal inversion) produces
   DataQualityStatus.INVALID and AnomalySeverity.INVALID_INPUT, NOT a behavioral anomaly.
3. Missing baseline produces AnomalySeverity.INSUFFICIENT_EVIDENCE, NOT confidently NORMAL.
4. Physical speed violations (e.g. >120 km/h) are categorized as PHYSICAL_INCONSISTENCY.
5. Inferred routes on closed roads are categorized as NETWORK_CONSTRAINT_INCONSISTENCY
   (incompatible network state), NOT suspicious driver behavior.
6. Forbidden overclaimed terminology ('loitering', 'unexpected stop', 'suspicious', 'criminal')
   is strictly replaced with precise, neutral behavioral descriptions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from schemas.anomaly_schema import (
    AnomalyEvidence,
    AnomalyResult,
    AnomalySeverity,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
    SignalCategory,
    SignalSeverity,
)
from schemas.normalized_trajectory_schema import NormalizedCandidateRoute, NormalizedTrajectory


class TrajectoryAnomalyDetector:
    """
    Detector for vehicle-level trajectory anomalies and data quality validation.
    """

    def __init__(
        self,
        baseline: Optional[MobilityBaseline] = None,
        max_physical_speed_kmh: float = 120.0,
        speed_tolerance_factor: float = 1.25,
    ) -> None:
        self.baseline = baseline or MobilityBaseline(source="unavailable")
        self.max_physical_speed_kmh = max_physical_speed_kmh
        self.speed_tolerance_factor = speed_tolerance_factor

    def evaluate_trajectory(
        self,
        trajectory: NormalizedTrajectory,
        road_graph: Optional[Any] = None,
    ) -> AnomalyResult:
        """
        Evaluate a NormalizedTrajectory and generate a consolidated AnomalyResult.
        """
        tid = trajectory.track_id
        orig = trajectory.origin_node
        dest = trajectory.destination_node
        t_start = trajectory.time_window_start
        t_end = trajectory.time_window_end

        # ---------------------------------------------------------------------
        # 1. INPUT VALIDATION & DATA QUALITY CHECK
        # ---------------------------------------------------------------------
        data_quality_issue = self._validate_input_integrity(trajectory)
        if data_quality_issue is not None:
            evidence = AnomalyEvidence(
                signal_type="data_quality_error",
                signal_category=SignalCategory.DATA_QUALITY,
                available=False,
                severity=SignalSeverity.EXTREME,
                explanation=data_quality_issue,
            )
            return AnomalyResult(
                anomaly_id=f"anomaly_veh_{tid}",
                entity_type="vehicle",
                entity_id=tid,
                data_quality_status=DataQualityStatus.INVALID,
                anomaly_types=["data_quality_error"],
                overall_score=None,
                severity=AnomalySeverity.INVALID_INPUT,
                evidence=[evidence],
                reliability=None,
                uncertainty=None,
                investigation_priority=InvestigationPriority.DATA_QUALITY_REVIEW,
                explanation=(
                    f"Data quality error for trajectory '{tid}': {data_quality_issue} "
                    f"Processing halted; not classified as a behavioral anomaly."
                ),
                timestamp_or_window={"start": t_start, "end": t_end},
                metadata={"origin": orig, "destination": dest},
            )

        # ---------------------------------------------------------------------
        # 2. EVALUATE CONSTRAINTS & SIGNALS
        # ---------------------------------------------------------------------
        evidence_list: List[AnomalyEvidence] = []
        triggered_types: List[str] = []

        # A. Physical Inconsistency (Excessive required speed)
        phys_evidence = self._check_physical_inconsistency(trajectory, road_graph)
        evidence_list.append(phys_evidence)
        if phys_evidence.signal_score >= 0.30:
            triggered_types.append(phys_evidence.signal_type)

        # B. Network Constraint Inconsistency (Traversing currently closed road)
        net_evidence = self._check_network_constraint(trajectory, road_graph)
        if net_evidence is not None:
            evidence_list.append(net_evidence)
            if net_evidence.signal_score >= 0.30:
                triggered_types.append(net_evidence.signal_type)

        # C. Travel-Time Deviation (vs. baseline)
        time_evidence = self._check_travel_time_deviation(trajectory)
        evidence_list.append(time_evidence)
        if time_evidence.signal_score >= 0.30:
            triggered_types.append(time_evidence.signal_type)

        # D. Route Corridor Deviation (vs. baseline)
        route_evidence = self._check_route_deviation(trajectory, road_graph)
        evidence_list.append(route_evidence)
        if route_evidence.signal_score >= 0.30:
            triggered_types.append(route_evidence.signal_type)

        # ---------------------------------------------------------------------
        # 3. EVIDENCE FUSION & SEVERITY DETERMINATION
        # ---------------------------------------------------------------------
        available_signals = [e for e in evidence_list if e.available]
        elevated_signals = [e for e in available_signals if e.signal_score >= 0.30]

        # Check if baselines were missing for all behavioral signals
        behavioral_available = any(
            e.available and e.signal_category == SignalCategory.BEHAVIORAL_ANOMALY
            for e in evidence_list
        )

        if not available_signals or (not behavioral_available and not elevated_signals):
            # No valid baseline existed to evaluate behavioral deviation
            overall_score = None
            severity = AnomalySeverity.INSUFFICIENT_EVIDENCE
            data_quality_status = DataQualityStatus.INCOMPLETE
        elif not elevated_signals:
            # Verified against baseline and found completely within normal limits
            overall_score = max(e.signal_score for e in available_signals)
            severity = AnomalySeverity.NORMAL
            data_quality_status = DataQualityStatus.VALID
        elif len(elevated_signals) == 1:
            overall_score = elevated_signals[0].signal_score
            severity = AnomalySeverity.classify(overall_score, has_evaluated_signals=True)
            data_quality_status = DataQualityStatus.VALID
        else:
            # Multi-signal fusion: dominant signal reinforced by co-occurring signals
            weights = {
                "physical_inconsistency": 1.5,
                "network_constraint_inconsistency": 1.2,
                "travel_time_deviation": 1.0,
                "route_deviation": 1.0,
            }
            sorted_elevated = sorted(elevated_signals, key=lambda e: e.signal_score, reverse=True)
            primary_score = sorted_elevated[0].signal_score
            reinforcement = sum(
                e.signal_score * 0.15 * (weights.get(e.signal_type, 1.0) / 1.5)
                for e in sorted_elevated[1:]
            )
            overall_score = min(1.0, primary_score + reinforcement)
            severity = AnomalySeverity.classify(overall_score, has_evaluated_signals=True)
            data_quality_status = DataQualityStatus.VALID

        # Extract Day 5 reliability and uncertainty metadata
        reliability = None
        uncertainty = None
        traj_meta = trajectory.metadata or {}
        if "reliability" in traj_meta:
            rel_val = traj_meta["reliability"]
            if isinstance(rel_val, dict):
                reliability = float(rel_val.get("overall_reliability", rel_val.get("tracking_reliability", rel_val.get("composite_reliability", rel_val.get("reliability", 0.5)))))
            else:
                reliability = float(rel_val)
        elif "tracking_reliability" in traj_meta:
            reliability = float(traj_meta["tracking_reliability"])
        elif "trajectory_reliability" in traj_meta:
            reliability = float(traj_meta["trajectory_reliability"])

        if "uncertainty" in traj_meta:
            unc_val = traj_meta["uncertainty"]
            if isinstance(unc_val, dict):
                uncertainty = float(unc_val.get("overall_uncertainty", unc_val.get("composite_uncertainty", unc_val.get("tracking_uncertainty", unc_val.get("uncertainty", 0.5)))))
            else:
                uncertainty = float(unc_val)
        elif "tracking_uncertainty" in traj_meta:
            uncertainty = float(traj_meta["tracking_uncertainty"])
        elif "trajectory_uncertainty" in traj_meta:
            uncertainty = float(traj_meta["trajectory_uncertainty"])

        # Determine operational review priority
        priority = InvestigationPriority.determine(
            severity, reliability, uncertainty, data_quality_status=data_quality_status
        )

        # Synthesize explanation
        explanation = self._synthesize_explanation(
            trajectory, triggered_types, overall_score, severity, priority, evidence_list, reliability
        )

        time_window = None
        if t_start is not None or t_end is not None:
            time_window = {"start": t_start, "end": t_end}

        return AnomalyResult(
            anomaly_id=f"anomaly_veh_{tid}",
            entity_type="vehicle",
            entity_id=tid,
            data_quality_status=data_quality_status,
            anomaly_types=triggered_types,
            overall_score=overall_score,
            severity=severity,
            evidence=evidence_list,
            reliability=reliability,
            uncertainty=uncertainty,
            investigation_priority=priority,
            explanation=explanation,
            timestamp_or_window=time_window,
            metadata={"origin": orig, "destination": dest},
        )

    def _validate_input_integrity(self, trajectory: NormalizedTrajectory) -> Optional[str]:
        """
        Validate input trajectory fields and temporal consistency.
        Returns error explanation if input is invalid/malformed, else None.
        """
        if not trajectory.track_id or not isinstance(trajectory.track_id, str):
            return "Missing or invalid track_id string."

        if not trajectory.candidate_routes or not isinstance(trajectory.candidate_routes, list):
            return "Missing candidate_routes list."

        t_start = trajectory.time_window_start
        t_end = trajectory.time_window_end

        if t_start is not None and t_end is not None:
            try:
                dt = float(t_end) - float(t_start)
                if dt < 0:
                    return f"Temporal inversion: end timestamp ({t_end}) precedes start timestamp ({t_start}) (delta_t: {dt:.1f}s)."
            except (ValueError, TypeError):
                return f"Non-numeric timestamp values encountered (start: {t_start!r}, end: {t_end!r})."

        return None

    def _check_physical_inconsistency(
        self,
        trajectory: NormalizedTrajectory,
        road_graph: Optional[Any],
    ) -> AnomalyEvidence:
        """
        Evaluate physical vehicle travel speed feasibility against physical limits.
        """
        t_start = trajectory.time_window_start
        t_end = trajectory.time_window_end

        if not trajectory.candidate_routes or t_start is None or t_end is None:
            return AnomalyEvidence(
                signal_type="physical_inconsistency",
                signal_category=SignalCategory.PHYSICAL_INCONSISTENCY,
                measured_value=0.0,
                baseline_value=0.0,
                threshold=0.0,
                signal_score=0.0,
                severity=SignalSeverity.NORMAL,
                available=False,
                explanation="Physical speed evaluation unavailable: incomplete route or timing information.",
            )

        try:
            dt = float(t_end) - float(t_start)
            if dt > 0:
                primary_route = max(trajectory.candidate_routes, key=lambda r: r.probability)
                dist_m = primary_route.metadata.get("distance_m")
                if dist_m is None and road_graph and hasattr(road_graph, "get_path_distance_m"):
                    dist_m = road_graph.get_path_distance_m(primary_route.nodes)

                if dist_m and dist_m > 0:
                    required_speed_kmh = (dist_m / dt) * 3.6
                    max_speed = self.baseline.tolerance_factors.get(
                        "max_physical_speed_kmh", self.max_physical_speed_kmh
                    )

                    if required_speed_kmh > max_speed:
                        return AnomalyEvidence(
                            signal_type="physical_inconsistency",
                            signal_category=SignalCategory.PHYSICAL_INCONSISTENCY,
                            measured_value=round(required_speed_kmh, 2),
                            baseline_value=max_speed,
                            threshold=max_speed,
                            signal_score=1.0,
                            severity=SignalSeverity.EXTREME,
                            available=True,
                            explanation=(
                                f"Physical speed violation: required travel speed of {required_speed_kmh:.1f} km/h "
                                f"over {dist_m:.1f}m in {dt:.1f}s exceeds maximum physical limit ({max_speed:.1f} km/h)."
                            ),
                            metadata={
                                "inconsistency_type": "excessive_speed",
                                "required_speed_kmh": round(required_speed_kmh, 2),
                                "distance_m": dist_m,
                                "delta_t": dt,
                            },
                        )
        except (ValueError, TypeError):
            pass

        return AnomalyEvidence(
            signal_type="physical_inconsistency",
            signal_category=SignalCategory.PHYSICAL_INCONSISTENCY,
            measured_value=0.0,
            baseline_value=0.0,
            threshold=0.0,
            signal_score=0.0,
            severity=SignalSeverity.NORMAL,
            available=True,
            explanation="Physical movement speed is consistent with vehicle physical capabilities.",
        )

    def _check_network_constraint(
        self,
        trajectory: NormalizedTrajectory,
        road_graph: Optional[Any],
    ) -> Optional[AnomalyEvidence]:
        """
        Evaluate whether the inferred route hypothesis traverses currently closed road segments.
        Explicitly labeled as a network-state incompatibility, NOT suspicious behavior.
        """
        if not road_graph or not hasattr(road_graph, "is_road_closed"):
            return None

        closed_edges = []
        for route in trajectory.candidate_routes:
            for edge in route.metadata.get("edges", []):
                if road_graph.is_road_closed(edge):
                    closed_edges.append(edge)

        if closed_edges:
            edges_str = ", ".join(set(closed_edges))
            return AnomalyEvidence(
                signal_type="network_constraint_inconsistency",
                signal_category=SignalCategory.NETWORK_CONSTRAINT_INCONSISTENCY,
                measured_value=float(len(closed_edges)),
                baseline_value=0.0,
                threshold=0.0,
                signal_score=0.70,
                severity=SignalSeverity.ELEVATED,
                available=True,
                explanation=(
                    f"The inferred route is incompatible with the current road-network state "
                    f"(traverses closed segment(s): {edges_str})."
                ),
                metadata={"closed_roads": list(set(closed_edges))},
            )

        return None

    def _check_travel_time_deviation(
        self,
        trajectory: NormalizedTrajectory,
    ) -> AnomalyEvidence:
        """
        Compare actual travel time against configured baseline for origin-destination pair.
        """
        orig = trajectory.origin_node
        dest = trajectory.destination_node
        t_start = trajectory.time_window_start
        t_end = trajectory.time_window_end

        if t_start is None or t_end is None:
            return AnomalyEvidence(
                signal_type="travel_time_deviation",
                signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation="Travel time evaluation unavailable: observation window missing start or end timestamp.",
            )

        try:
            delta_t = float(t_end) - float(t_start)
        except (ValueError, TypeError):
            return AnomalyEvidence(
                signal_type="travel_time_deviation",
                signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation="Travel time evaluation unavailable: invalid non-numeric timestamp values.",
            )

        expected_t = self.baseline.get_expected_travel_time(orig, dest)
        if expected_t is None or expected_t <= 0:
            return AnomalyEvidence(
                signal_type="travel_time_deviation",
                signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
                measured_value=delta_t,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation=f"Travel time baseline unavailable: no configured reference travel time for OD pair {orig}->{dest}.",
            )

        # Compute normalized deviation ratio
        deviation_ratio = abs(delta_t - expected_t) / max(expected_t, 1.0)
        elevated_thresh = self.baseline.tolerance_factors.get("travel_time_elevated_ratio", 0.40)
        high_thresh = self.baseline.tolerance_factors.get("travel_time_high_ratio", 0.80)
        extreme_thresh = self.baseline.tolerance_factors.get("travel_time_extreme_ratio", 1.50)

        if deviation_ratio < elevated_thresh:
            signal_score = 0.0
            severity = SignalSeverity.NORMAL
            explanation = (
                f"Travel time ({delta_t:.1f}s) is consistent with baseline ({expected_t:.1f}s)."
            )
        elif deviation_ratio < high_thresh:
            fraction = (deviation_ratio - elevated_thresh) / max(high_thresh - elevated_thresh, 0.01)
            signal_score = 0.35 + 0.25 * fraction
            severity = SignalSeverity.ELEVATED
            ratio_mult = delta_t / expected_t
            explanation = (
                f"Elevated travel time: {delta_t:.1f}s is {ratio_mult:.2f}x expected baseline ({expected_t:.1f}s)."
            )
        elif deviation_ratio < extreme_thresh:
            fraction = (deviation_ratio - high_thresh) / max(extreme_thresh - high_thresh, 0.01)
            signal_score = 0.60 + 0.25 * fraction
            severity = SignalSeverity.HIGH
            ratio_mult = delta_t / expected_t
            explanation = (
                f"Substantial travel-time deviation: {delta_t:.1f}s is {ratio_mult:.2f}x expected baseline ({expected_t:.1f}s)."
            )
        else:
            fraction = min(1.0, (deviation_ratio - extreme_thresh) / max(extreme_thresh, 1.0))
            signal_score = 0.85 + 0.15 * fraction
            severity = SignalSeverity.EXTREME
            ratio_mult = delta_t / expected_t
            explanation = (
                f"Extreme travel-time deviation: {delta_t:.1f}s is {ratio_mult:.2f}x expected baseline ({expected_t:.1f}s)."
            )

        return AnomalyEvidence(
            signal_type="travel_time_deviation",
            signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
            measured_value=delta_t,
            baseline_value=expected_t,
            threshold=elevated_thresh * expected_t,
            signal_score=signal_score,
            severity=severity,
            available=True,
            explanation=explanation,
            metadata={"deviation_ratio": round(deviation_ratio, 4), "origin": orig, "destination": dest},
        )

    def _check_route_deviation(
        self,
        trajectory: NormalizedTrajectory,
        road_graph: Optional[Any],
    ) -> AnomalyEvidence:
        """
        Compare actual traversed route corridor against configured expected route corridor.
        """
        orig = trajectory.origin_node
        dest = trajectory.destination_node

        expected_route = self.baseline.get_expected_route(orig, dest)
        if expected_route is None or len(expected_route) < 2:
            return AnomalyEvidence(
                signal_type="route_deviation",
                signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation=f"Route corridor baseline unavailable: no expected route configured for OD pair {orig}->{dest}.",
            )

        if not trajectory.candidate_routes:
            return AnomalyEvidence(
                signal_type="route_deviation",
                signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation="Route deviation evaluation unavailable: trajectory contains no candidate routes.",
            )

        # Identify primary route (highest relative likelihood)
        primary_route = max(trajectory.candidate_routes, key=lambda r: r.probability)
        actual_nodes = list(primary_route.nodes)

        # Jaccard node dissimilarity
        set_act = set(actual_nodes)
        set_exp = set(expected_route)
        intersection = len(set_act & set_exp)
        union = len(set_act | set_exp)
        dissimilarity = 1.0 - (intersection / max(union, 1))

        dissimilarity_thresh = self.baseline.tolerance_factors.get("route_deviation_dissimilarity", 0.40)

        # Explicit route ambiguity preservation
        is_ambiguous = bool(trajectory.metadata.get("is_ambiguous", False))
        if len(trajectory.candidate_routes) > 1:
            probs = sorted([r.probability for r in trajectory.candidate_routes], reverse=True)
            if len(probs) >= 2 and (probs[0] - probs[1]) < 0.20:
                is_ambiguous = True

        if dissimilarity < dissimilarity_thresh:
            signal_score = 0.0
            severity = SignalSeverity.NORMAL
            explanation = (
                f"Traversed route corridor matches expected corridor {expected_route} "
                f"(overlap: {intersection}/{union} nodes)."
            )
        else:
            raw_score = min(1.0, 0.40 + 0.60 * ((dissimilarity - dissimilarity_thresh) / max(1.0 - dissimilarity_thresh, 0.1)))
            if is_ambiguous:
                # Dampen confidence and explicitly document ambiguity
                signal_score = max(0.30, raw_score * 0.70)
                severity = SignalSeverity.ELEVATED
                explanation = (
                    f"Route corridor differs from configured expected {expected_route} (dissimilarity: {dissimilarity:.2f}), "
                    f"but route inference remains ambiguous across {len(trajectory.candidate_routes)} candidate hypotheses "
                    f"(primary route probability: {primary_route.probability:.2f})."
                )
            else:
                signal_score = raw_score
                severity = SignalSeverity.HIGH if signal_score >= 0.65 else SignalSeverity.ELEVATED
                explanation = (
                    f"Route corridor deviation: traversed corridor {actual_nodes} differs from expected "
                    f"{expected_route} (dissimilarity: {dissimilarity:.2f})."
                )

        return AnomalyEvidence(
            signal_type="route_deviation",
            signal_category=SignalCategory.BEHAVIORAL_ANOMALY,
            measured_value=round(dissimilarity, 4),
            baseline_value=0.0,
            threshold=dissimilarity_thresh,
            signal_score=signal_score,
            severity=severity,
            available=True,
            explanation=explanation,
            metadata={
                "actual_nodes": actual_nodes,
                "expected_nodes": expected_route,
                "dissimilarity": round(dissimilarity, 4),
                "is_ambiguous": is_ambiguous,
            },
        )

    def _synthesize_explanation(
        self,
        trajectory: NormalizedTrajectory,
        triggered_types: List[str],
        overall_score: Optional[float],
        severity: AnomalySeverity,
        priority: InvestigationPriority,
        evidence_list: List[AnomalyEvidence],
        reliability: Optional[float],
    ) -> str:
        tid = trajectory.track_id
        orig = trajectory.origin_node
        dest = trajectory.destination_node

        if severity == AnomalySeverity.INSUFFICIENT_EVIDENCE:
            return (
                f"Cannot evaluate behavioral anomaly for trajectory '{tid}' (corridor {orig}->{dest}): "
                f"no configured baseline travel time or route corridor is available. "
                f"Evidence is insufficient for anomaly determination."
            )

        if severity == AnomalySeverity.NORMAL:
            return (
                f"Trajectory '{tid}' (corridor {orig}->{dest}) matches baseline expectations and physical constraints; "
                f"no behavioral anomaly detected (overall deviation score: {overall_score:.2f})."
            )

        reasons = [e.explanation for e in evidence_list if e.signal_score >= 0.30]
        reasons_text = "; ".join(reasons) if reasons else "Elevated deviation across available signals."

        rel_str = f"evidence reliability is {reliability:.2f}" if reliability is not None else "reliability uncalibrated"
        downgrade_note = ""
        if severity in (AnomalySeverity.HIGH_PRIORITY, AnomalySeverity.INVESTIGATE) and priority == InvestigationPriority.WATCH:
            downgrade_note = " (Note: priority adjusted to WATCH due to low sensor evidence reliability)."

        score_str = f"{overall_score:.2f}" if overall_score is not None else "N/A"

        return (
            f"Investigation Candidate '{tid}' on corridor {orig}->{dest} (Severity: {severity.value.upper()}, "
            f"Priority: {priority.value.upper()}{downgrade_note}): Overall anomaly deviation score is {score_str} "
            f"supported by {len(triggered_types)} signal(s). Reasons: {reasons_text} [{rel_str}]."
        )
