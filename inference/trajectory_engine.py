"""
Trajectory Reconstruction Engine for UrbanTrack AI (Day 3).
Provides candidate route generation, road-aware feasibility, explainable route scoring,
uncertainty preservation, and multi-observation vehicle trajectory inference.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory
from .road_graph import RoadGraph


def evaluate_route_feasibility_and_score(
    path_data: Dict[str, Any],
    delta_t_seconds: Optional[float],
    shortest_distance_m: float,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[float], float, float]:
    """
    Evaluate road-aware feasibility and compute an explainable unnormalized route score.

    Args:
        path_data: Path dictionary containing edges, nodes, distance_m, speed_limit_kmh, etc.
        delta_t_seconds: Observed time difference between observations (t_b - t_a), or None if unavailable.
        shortest_distance_m: Distance of the shortest candidate path in the network.
        config: Optional configuration settings (e.g. speed tolerance, weightings).

    Returns:
        Tuple[bool, str, Optional[float], float, float]:
            (is_feasible, status_code, required_speed_kmh, raw_score, estimated_travel_time_s)
    """
    config = config or {}
    dist_m = float(path_data["distance_m"])
    speed_limit_kmh = float(path_data.get("speed_limit_kmh", 50.0))
    est_travel_time_s = float(path_data.get("estimated_travel_time_s", dist_m / (40.0 / 3.6)))
    min_travel_time_s = float(path_data.get("min_travel_time_s", dist_m / (speed_limit_kmh / 3.6)))

    speed_tolerance_factor = float(config.get("speed_tolerance_factor", 1.25))  # Allow 25% over speed limit
    max_absolute_speed_kmh = float(config.get("max_absolute_speed_kmh", 120.0))

    # When temporal evidence is unavailable:
    if delta_t_seconds is None:
        # Route cannot be verified temporally; retained as feasible_temporally_unverified
        # based on valid road connectivity and topology.
        # Scored purely on distance efficiency among candidates without fake speed.
        if dist_m > 0:
            dist_score = min(1.0, shortest_distance_m / dist_m)
        else:
            dist_score = 1.0
        return (True, "feasible_temporally_unverified", None, dist_score, est_travel_time_s)

    # 1. Temporal Inversion (t_b < t_a)
    if delta_t_seconds < 0:
        return (False, "temporal_inversion", 0.0, 0.0, est_travel_time_s)

    # 2. Simultaneous Observations (delta_t == 0)
    if delta_t_seconds == 0:
        if dist_m == 0:
            return (True, "stationary_same_instant", 0.0, 1.0, 0.0)
        else:
            return (False, "impossible_simultaneous_distinct_locations", float("inf"), 0.0, est_travel_time_s)

    # 3. Calculate Required Travel Speed
    required_speed_kmh = (dist_m / delta_t_seconds) * 3.6

    # 4. Check Road-Aware Speed Limits & Physical Speed Feasibility
    effective_max_speed = min(speed_limit_kmh * speed_tolerance_factor, max_absolute_speed_kmh)

    if required_speed_kmh > effective_max_speed:
        # Travel speed physically impossible or exceeds road limit
        return (False, "speed_limit_exceeded", required_speed_kmh, 0.0, est_travel_time_s)

    # 5. Feasible Route Scoring (Explainable Composite Metric)
    # A. Speed compatibility score: evaluates closeness to free-flow expected speed
    expected_speed_kmh = float(path_data.get("expected_speed_kmh", speed_limit_kmh * 0.8))
    if required_speed_kmh <= expected_speed_kmh:
        # Lower speed than free-flow is plausible due to traffic/signals/stops
        # Slow exponential decay for very low speeds
        ratio = required_speed_kmh / max(expected_speed_kmh, 1.0)
        speed_score = max(0.2, math.sqrt(ratio))
    else:
        # Speed between expected and speed limit tolerance
        margin = max(effective_max_speed - expected_speed_kmh, 1.0)
        speed_score = max(0.1, 1.0 - 0.7 * ((required_speed_kmh - expected_speed_kmh) / margin))

    # B. Distance efficiency score (relative to shortest path)
    if dist_m > 0:
        dist_score = min(1.0, shortest_distance_m / dist_m)
    else:
        dist_score = 1.0

    # Composite unnormalized raw score
    raw_score = 0.65 * speed_score + 0.35 * dist_score

    return (True, "feasible", required_speed_kmh, raw_score, est_travel_time_s)


def reconstruct_trajectory_segment(
    obs_a: Optional[Union[Observation, Dict[str, Any]]] = None,
    obs_b: Optional[Union[Observation, Dict[str, Any]]] = None,
    road_graph: Optional[RoadGraph] = None,
    identity_id: str = "vehicle_candidate",
    config: Optional[Dict[str, Any]] = None,
    max_paths: Optional[int] = None,
    *,
    start_obs: Optional[Union[Observation, Dict[str, Any]]] = None,
    end_obs: Optional[Union[Observation, Dict[str, Any]]] = None,
) -> TrajectorySegment:
    """
    Reconstruct candidate trajectories between two sequential vehicle observations.

    Generates top-K alternative routes, checks road-aware feasibility, computes
    explainable relative likelihoods, and preserves routing uncertainty.

    Args:
        obs_a: Origin observation (Observation instance or dict).
        obs_b: Destination observation (Observation instance or dict).
        road_graph: RoadGraph spatial network instance.
        identity_id: Vehicle identity identifier.
        config: Optional inference parameters (max_paths, ambiguity_threshold, etc.).
        max_paths: Optional override for maximum candidate paths to explore.
        start_obs: Optional alias for obs_a.
        end_obs: Optional alias for obs_b.

    Returns:
        TrajectorySegment: Segment hypothesis containing ranked candidate routes and uncertainty.
    """
    obs_a = obs_a or start_obs
    obs_b = obs_b or end_obs
    if obs_a is None or obs_b is None:
        raise ValueError("Both origin (obs_a/start_obs) and destination (obs_b/end_obs) observations must be provided.")
    if road_graph is None:
        raise ValueError("road_graph must be provided.")

    config = dict(config or {})
    if max_paths is not None:
        config["max_candidate_paths"] = max_paths
    max_k = int(config.get("max_candidate_paths", 5))
    ambiguity_threshold = float(config.get("ambiguity_threshold", 0.15))

    # Helper to extract fields from Observation or Dict
    def _extract(obs: Union[Observation, Dict[str, Any]]) -> Tuple[str, str, float, Optional[float], Optional[float]]:
        if isinstance(obs, Observation):
            return (obs.observation_id, obs.camera_id, obs.timestamp_seconds, obs.latitude, obs.longitude)
        return (
            str(obs.get("observation_id", "obs_a")),
            str(obs.get("camera_id", "")),
            float(obs.get("timestamp_seconds", obs.get("timestamp", 0.0))),
            obs.get("latitude"),
            obs.get("longitude"),
        )

    id_a, cam_a, t_a, lat_a, lon_a = _extract(obs_a)
    id_b, cam_b, t_b, lat_b, lon_b = _extract(obs_b)

    from .temporal import check_temporal_comparability

    cam_meta_source = None
    if road_graph and hasattr(road_graph, "metadata") and road_graph.metadata:
        cam_meta_source = road_graph.metadata
    if config and "camera_metadata" in config:
        cam_meta_source = config["camera_metadata"]

    t_comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=cam_meta_source)
    if t_comp["comparable"]:
        delta_t = t_comp["delta_seconds"]
    elif t_comp["status"] == "invalid_negative_time":
        delta_t = -1.0
    else:
        delta_t = None

    # Associate cameras with road network junctions
    node_a = road_graph.associate_camera(cam_a, latitude=lat_a, longitude=lon_a)
    node_b = road_graph.associate_camera(cam_b, latitude=lat_b, longitude=lon_b)

    segment = TrajectorySegment(
        segment_id=f"{id_a}->{id_b}",
        identity_id=identity_id,
        start_observation_id=id_a,
        start_camera_id=cam_a,
        start_timestamp=t_a,
        end_observation_id=id_b,
        end_camera_id=cam_b,
        end_timestamp=t_b,
        time_difference_seconds=delta_t,
        start_node_id=node_a,
        end_node_id=node_b,
        temporal_evidence=t_comp,
    )

    # Check Negative Time Difference (Reversed Timestamps)
    if t_comp["status"] == "invalid_negative_time" or (delta_t is not None and delta_t < 0):
        segment.status = "infeasible"
        segment.is_ambiguous = False
        segment.ambiguity_reason = (
            f"Negative time difference: Observation timestamps are reversed ({t_comp.get('reason')})."
        )
        return segment

    # Check Camera Association Failures
    if node_a is None or node_b is None:
        unassoc = []
        if node_a is None:
            unassoc.append(f"start camera '{cam_a}'")
        if node_b is None:
            unassoc.append(f"destination camera '{cam_b}'")
        segment.status = "unassociated_camera"
        segment.is_ambiguous = True
        segment.ambiguity_reason = f"Camera could not be associated with road network: {', '.join(unassoc)}."
        return segment

    # Check Stationary Observations (Same Node / Camera)
    if node_a == node_b:
        segment.status = "success"
        segment.most_likely_route = []
        c_route = CandidateRoute(
            route_id="route_stationary",
            edges=[],
            nodes=[node_a],
            distance_meters=0.0,
            estimated_travel_time_seconds=0.0,
            min_travel_time_seconds=0.0,
            required_speed_kmh=0.0 if delta_t is not None else None,
            speed_limit_kmh=50.0,
            feasible=True,
            feasibility_status="stationary",
            estimated_likelihood=1.0,
            raw_score=1.0,
            explanation=(
                f"Stationary vehicle observed at same camera/junction (duration Δt = {delta_t:.1f}s)."
                if delta_t is not None
                else "Stationary vehicle observed at same junction (temporal duration unavailable)."
            ),
            temporal_evidence=t_comp,
        )
        segment.candidate_routes = [c_route]
        segment.confidence = 1.0
        segment.is_ambiguous = False
        segment.ambiguity_reason = None
        return segment

    # Find Candidate Paths
    candidate_paths = road_graph.find_candidate_paths(node_a, node_b, max_paths=max_k)

    if not candidate_paths:
        segment.status = "no_path"
        segment.is_ambiguous = False
        segment.ambiguity_reason = f"No connected road network path found between '{node_a}' and '{node_b}'."
        return segment

    shortest_dist = candidate_paths[0]["distance_m"]
    evaluated_routes: List[CandidateRoute] = []

    for idx, path in enumerate(candidate_paths):
        feasible, status, req_speed, raw_score, est_time = evaluate_route_feasibility_and_score(
            path, delta_t, shortest_dist, config=config
        )

        speed_limit = float(path.get("speed_limit_kmh", 50.0))
        min_time = float(path.get("min_travel_time_s", 0.0))
        dist_m = float(path["distance_m"])

        # Construct explainable rationale
        if status == "temporal_inversion":
            expl = f"Route rejected: Observation timestamps reversed ({t_comp.get('reason')})."
        elif status == "speed_limit_exceeded":
            expl = (
                f"Route infeasible: Required speed of {req_speed:.1f} km/h over {dist_m:.1f}m in {delta_t:.1f}s "
                f"exceeds road speed limit ({speed_limit:.1f} km/h)."
            )
        elif status == "stationary_same_instant":
            expl = "Vehicle observed at same junction and instant."
        elif status == "feasible_temporally_unverified":
            expl = (
                f"Route temporally unverified ({dist_m:.1f}m): Road connectivity and topology feasible, "
                f"but cross-camera temporal evidence is unavailable ({t_comp.get('reason')})."
            )
        else:
            expl = (
                f"Feasible route ({dist_m:.1f}m): Required speed {req_speed:.1f} km/h is compatible with "
                f"road speed limit ({speed_limit:.1f} km/h)."
            )

        c_route = CandidateRoute(
            route_id=f"route_{idx + 1:02d}",
            edges=path["edges"],
            nodes=path["nodes"],
            distance_meters=dist_m,
            estimated_travel_time_seconds=est_time,
            min_travel_time_seconds=min_time,
            required_speed_kmh=req_speed,
            speed_limit_kmh=speed_limit,
            feasible=feasible,
            feasibility_status=status,
            estimated_likelihood=0.0,  # Will normalize below
            raw_score=raw_score,
            explanation=expl,
            temporal_evidence=t_comp,
        )
        evaluated_routes.append(c_route)

    # Normalize Relative Likelihoods Across Candidate Routes
    # (Documented: Relative estimated likelihoods, NOT calibrated statistical posterior probabilities)
    feasible_candidates = [r for r in evaluated_routes if r.feasible]
    total_raw = sum(r.raw_score for r in feasible_candidates)

    if total_raw > 0:
        for r in feasible_candidates:
            r.estimated_likelihood = round(r.raw_score / total_raw, 4)
    elif feasible_candidates:
        # Equal prior if raw scores are zero
        equal_prob = round(1.0 / len(feasible_candidates), 4)
        for r in feasible_candidates:
            r.estimated_likelihood = equal_prob

    # Rank candidates descending by estimated likelihood, then distance
    evaluated_routes.sort(key=lambda r: (r.feasible, r.estimated_likelihood, -r.distance_meters), reverse=True)
    segment.candidate_routes = evaluated_routes

    if feasible_candidates:
        segment.status = "success"
        segment.most_likely_route = evaluated_routes[0].edges
        segment.confidence = evaluated_routes[0].estimated_likelihood

        # Check for Routing Ambiguity (Uncertainty Preservation)
        if len(feasible_candidates) > 1:
            diff = evaluated_routes[0].estimated_likelihood - evaluated_routes[1].estimated_likelihood
            if diff < ambiguity_threshold:
                segment.is_ambiguous = True
                segment.ambiguity_reason = (
                    f"Trajectory is ambiguous between Route 1 ({evaluated_routes[0].estimated_likelihood * 100:.1f}%) "
                    f"and Route 2 ({evaluated_routes[1].estimated_likelihood * 100:.1f}%)."
                )
    else:
        segment.status = "infeasible"
        segment.most_likely_route = None
        segment.confidence = 0.0
        segment.is_ambiguous = False
        segment.ambiguity_reason = "All candidate routes between observations are physically infeasible."

    # Day 5: Evaluate endpoint observation reliabilities and propagate uncertainty
    from .reliability_engine import evaluate_observation_reliability, propagate_trajectory_uncertainty

    obs_rel_a = evaluate_observation_reliability(obs_a, config=config)
    obs_rel_b = evaluate_observation_reliability(obs_b, config=config)
    traj_rel = propagate_trajectory_uncertainty(
        trajectory_id=segment.segment_id,
        obs_rel_a=obs_rel_a,
        obs_rel_b=obs_rel_b,
        candidate_routes=evaluated_routes,
        is_gap=getattr(segment, "is_gap", False),
        unobserved_intermediate_nodes=getattr(segment, "unobserved_intermediate_nodes", []),
        gap_duration_seconds=segment.time_difference_seconds if getattr(segment, "is_gap", False) else None,
        config=config,
    )
    segment.reliability = traj_rel.to_dict()
    segment.uncertainty = traj_rel.to_dict()

    return segment


def reconstruct_identity_trajectory(
    identity_data: Dict[str, Any],
    road_graph: RoadGraph,
    config: Optional[Dict[str, Any]] = None,
) -> VehicleTrajectory:
    """
    Reconstruct full multi-segment trajectory for a vehicle identity cluster.

    Accepts output produced by Day-2 Identity Graph (get_candidate_identities),
    chains sequential observations, evaluates all intermediate transitions,
    and produces the Day 3 Output Contract schema.

    Args:
        identity_data: Dictionary representing candidate vehicle identity.
        road_graph: Spatial RoadGraph instance.
        config: Optional configuration dictionary.

    Returns:
        VehicleTrajectory: Full multi-observation vehicle trajectory hypothesis.
    """
    config = config or {}
    ident_id = identity_data.get("identity_id", identity_data.get("candidate_vehicle_id", "vehicle_001"))

    # Extract member observations (handles member_observations or observations)
    raw_obs_list = identity_data.get("member_observations") or identity_data.get("observations") or []

    # Sort observations chronologically
    sorted_obs = sorted(
        raw_obs_list,
        key=lambda o: float(o.timestamp_seconds if isinstance(o, Observation) else o.get("timestamp_seconds", o.get("timestamp", 0.0))),
    )

    n_obs = len(sorted_obs)
    if n_obs == 0:
        return VehicleTrajectory(
            identity_id=ident_id,
            observations_count=0,
            cameras_visited=[],
            start_timestamp=0.0,
            end_timestamp=0.0,
            total_time_seconds=0.0,
            total_distance_meters=0.0,
            overall_confidence=0.0,
            is_ambiguous=False,
        )

    cameras = list(dict.fromkeys([str(o.camera_id if isinstance(o, Observation) else o.get("camera_id", "")) for o in sorted_obs]))
    t_start = float(sorted_obs[0].timestamp_seconds if isinstance(sorted_obs[0], Observation) else sorted_obs[0].get("timestamp_seconds", sorted_obs[0].get("timestamp", 0.0)))
    t_end = float(sorted_obs[-1].timestamp_seconds if isinstance(sorted_obs[-1], Observation) else sorted_obs[-1].get("timestamp_seconds", sorted_obs[-1].get("timestamp", 0.0)))
    total_time = t_end - t_start

    # Single observation case (stationary / singleton)
    if n_obs == 1:
        cam_single = cameras[0] if cameras else ""
        lat_single = sorted_obs[0].latitude if isinstance(sorted_obs[0], Observation) else sorted_obs[0].get("latitude")
        lon_single = sorted_obs[0].longitude if isinstance(sorted_obs[0], Observation) else sorted_obs[0].get("longitude")
        node_single = road_graph.associate_camera(
            cam_single,
            latitude=lat_single,
            longitude=lon_single,
        )
        return VehicleTrajectory(
            identity_id=ident_id,
            observations_count=1,
            cameras_visited=cameras,
            start_timestamp=t_start,
            end_timestamp=t_end,
            total_time_seconds=0.0,
            total_distance_meters=0.0,
            overall_confidence=1.0,
            is_ambiguous=False,
            complete_route_edges=[],
            complete_route_nodes=[node_single] if node_single else [],
        )

    # Reconstruct segments pairwise
    segments: List[TrajectorySegment] = []
    total_dist = 0.0
    total_time = 0.0
    has_unavailable_time = False
    cameras: List[str] = []
    conf_scores: List[float] = []
    any_ambiguous = False
    all_route_edges: List[str] = []
    all_route_nodes: List[str] = []

    for i in range(n_obs - 1):
        obs_1 = sorted_obs[i]
        obs_2 = sorted_obs[i + 1]

        seg = reconstruct_trajectory_segment(
            obs_a=obs_1,
            obs_b=obs_2,
            road_graph=road_graph,
            identity_id=ident_id,
            config=config,
        )
        segments.append(seg)

        # Accumulate metrics
        if seg.time_difference_seconds is not None:
            total_time += seg.time_difference_seconds
        else:
            has_unavailable_time = True

        if seg.start_camera_id and seg.start_camera_id not in cameras:
            cameras.append(seg.start_camera_id)
        if seg.end_camera_id and seg.end_camera_id not in cameras:
            cameras.append(seg.end_camera_id)

        if seg.is_ambiguous:
            any_ambiguous = True

        if seg.most_likely_route is not None:
            # Avoid duplicate node transitions
            if seg.candidate_routes and seg.candidate_routes[0].nodes:
                seg_nodes = seg.candidate_routes[0].nodes
                if not all_route_nodes:
                    all_route_nodes.extend(seg_nodes)
                else:
                    all_route_nodes.extend(seg_nodes[1:])
            all_route_edges.extend(seg.most_likely_route)
            total_dist += (seg.candidate_routes[0].distance_meters if seg.candidate_routes else 0.0)
            conf_scores.append(seg.confidence)
        else:
            conf_scores.append(0.0)

    # Overall trajectory confidence (geometric mean of segment confidences)
    if conf_scores and all(c > 0 for c in conf_scores):
        log_sum = sum(math.log(c) for c in conf_scores)
        overall_conf = math.exp(log_sum / len(conf_scores))
    elif conf_scores:
        overall_conf = sum(conf_scores) / len(conf_scores)
    else:
        overall_conf = 0.0

    # Overall reliability and uncertainty
    if segments:
        avg_rel = sum((s.reliability or {}).get("overall_reliability", 0.80) for s in segments) / len(segments)
        avg_unc = round(1.0 - avg_rel, 4)
        traj_reliability = {
            "overall_reliability": round(avg_rel, 4),
            "overall_uncertainty": avg_unc,
            "segments_count": len(segments),
        }
        traj_uncertainty = {
            "overall_uncertainty": avg_unc,
            "score": avg_unc,
            "level": "low" if avg_unc <= 0.25 else ("moderate" if avg_unc <= 0.50 else ("high" if avg_unc <= 0.75 else "critical")),
        }
    else:
        traj_reliability = {"overall_reliability": 1.0, "overall_uncertainty": 0.0}
        traj_uncertainty = {"overall_uncertainty": 0.0, "score": 0.0, "level": "low"}

    final_total_time = None if has_unavailable_time else total_time

    return VehicleTrajectory(
        identity_id=ident_id,
        observations_count=n_obs,
        cameras_visited=cameras,
        start_timestamp=t_start,
        end_timestamp=t_end,
        total_time_seconds=final_total_time,
        total_distance_meters=total_dist,
        overall_confidence=overall_conf,
        is_ambiguous=any_ambiguous,
        segments=segments,
        complete_route_edges=all_route_edges,
        complete_route_nodes=all_route_nodes,
        reliability=traj_reliability,
        uncertainty=traj_uncertainty,
    )
