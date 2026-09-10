"""
Day 4 Sparse / Missing-Camera Trajectory Inference Engine for UrbanTrack AI.

Infers plausible hidden vehicle trajectories across unobserved intervals (gaps)
between sightings of the same cross-camera vehicle identity.

CRITICAL PRINCIPLES:
1. Zero Observation Fabrication: NEVER synthesize fake observation records for unobserved cameras.
2. Unobserved Interval Representation: Represent gaps as observed endpoints + unobserved interval
   + candidate hidden routes + relative estimated likelihoods + ambiguity.
3. Reuse Day-3 Machinery: Leverage RoadGraph, bounded loop-free path search, directed road constraints,
   closed road avoidance, and road-aware temporal feasibility scoring.
4. Distinguish Missing Intermediate Sightings from Sensor Failure: State only what the evidence supports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from schemas.gap_schema import SparseObservationGap
from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory
from .road_graph import RoadGraph
from .trajectory_engine import (
    evaluate_route_feasibility_and_score,
    reconstruct_trajectory_segment,
)


def _extract_obs_info(obs: Union[Observation, Dict[str, Any]]) -> Tuple[str, str, float, Optional[float], Optional[float]]:
    """Extract standard tuple (id, camera_id, timestamp_seconds, lat, lon) from Observation or dict."""
    if isinstance(obs, Observation):
        return (obs.observation_id, obs.camera_id, obs.timestamp_seconds, obs.latitude, obs.longitude)
    return (
        str(obs.get("observation_id", "obs_unknown")),
        str(obs.get("camera_id", "")),
        float(obs.get("timestamp_seconds", obs.get("timestamp", 0.0))),
        obs.get("latitude"),
        obs.get("longitude"),
    )


def detect_observation_gaps(
    observations: List[Union[Observation, Dict[str, Any]]],
    road_graph: RoadGraph,
) -> List[Dict[str, Any]]:
    """
    Analyze a sequence of observations belonging to an identity and detect
    whether unobserved movement intervals (gaps) exist between sightings.

    A gap is defined when two sightings have a positive time difference and
    their topological connection requires traversing unobserved intermediate
    junctions or roads.

    Args:
        observations: Chronological list of observations.
        road_graph: RoadGraph spatial network instance.

    Returns:
        List[Dict[str, Any]]: List of transition analysis dictionaries describing
        whether each interval is a direct single-hop, stationary, or unobserved gap.
    """
    if len(observations) < 2:
        return []

    sorted_obs = sorted(
        observations,
        key=lambda o: _extract_obs_info(o)[2],
    )

    gap_analysis = []

    from .temporal import check_temporal_comparability

    cam_meta = road_graph.metadata if road_graph and hasattr(road_graph, "metadata") else None

    for i in range(len(sorted_obs) - 1):
        id_a, cam_a, t_a, lat_a, lon_a = _extract_obs_info(sorted_obs[i])
        id_b, cam_b, t_b, lat_b, lon_b = _extract_obs_info(sorted_obs[i + 1])

        t_comp = check_temporal_comparability(sorted_obs[i], sorted_obs[i + 1], camera_metadata=cam_meta)
        if t_comp["comparable"]:
            delta_t = t_comp["delta_seconds"]
        elif t_comp["status"] == "invalid_negative_time":
            delta_t = -1.0
        else:
            delta_t = None

        node_a = road_graph.associate_camera(cam_a, latitude=lat_a, longitude=lon_a)
        node_b = road_graph.associate_camera(cam_b, latitude=lat_b, longitude=lon_b)

        if delta_t is not None and delta_t < 0:
            classification = "invalid_time_interval"
            is_gap = False
        elif node_a is None or node_b is None:
            classification = "unavailable_camera_evidence"
            is_gap = True
        elif node_a == node_b:
            classification = "stationary_same_location"
            is_gap = False
        else:
            # Check shortest path edge count
            paths = road_graph.find_candidate_paths(node_a, node_b, max_paths=1)
            if not paths:
                classification = "disconnected_network"
                is_gap = True
            elif len(paths[0]["edges"]) > 1:
                # Traverses at least one intermediate junction
                classification = "missing_intermediate_observations"
                is_gap = True
            else:
                # Direct single road edge connecting node_a -> node_b
                classification = "direct_adjacent_corridor"
                is_gap = False

        gap_analysis.append({
            "index": i,
            "start_observation_id": id_a,
            "start_camera_id": cam_a,
            "start_node_id": node_a,
            "start_timestamp": t_a,
            "end_observation_id": id_b,
            "end_camera_id": cam_b,
            "end_node_id": node_b,
            "end_timestamp": t_b,
            "gap_duration_seconds": round(delta_t, 2) if delta_t is not None else None,
            "is_gap": is_gap,
            "classification": classification,
            "temporal_evidence": t_comp,
        })

    return gap_analysis


def infer_sparse_gap(
    obs_a: Union[Observation, Dict[str, Any]],
    obs_b: Union[Observation, Dict[str, Any]],
    road_graph: RoadGraph,
    identity_id: str = "vehicle_candidate",
    max_paths: int = 5,
    config: Optional[Dict[str, Any]] = None,
) -> SparseObservationGap:
    """
    Infer candidate hidden routes through an unobserved movement interval (gap).

    Given two observations belonging to the same cross-camera vehicle identity:
    1. Determines graph origin and destination nodes.
    2. Identifies unobserved interval duration (Δt = T_b - T_a).
    3. Reuses RoadGraph bounded candidate route generation.
    4. Evaluates temporal and road-aware feasibility for each candidate route.
    5. Rejects physically impossible routes (speed limits, closed roads, directed edges).
    6. Normalizes relative estimated likelihoods across feasible candidates.
    7. Preserves uncertainty and surfaces ambiguity when alternative routes are close.

    CRITICAL: Does NOT fabricate observation records for intermediate unobserved cameras.

    Args:
        obs_a: Origin sighting.
        obs_b: Later sighting.
        road_graph: RoadGraph instance (e.g. city_network.json).
        identity_id: Cross-camera vehicle identity ID.
        max_paths: Maximum candidate paths to explore.
        config: Optional configuration dictionary.

    Returns:
        SparseObservationGap: Explicit model of the unobserved movement interval.
    """
    config = dict(config or {})
    ambiguity_threshold = float(config.get("ambiguity_threshold", 0.15))

    id_a, cam_a, t_a, lat_a, lon_a = _extract_obs_info(obs_a)
    id_b, cam_b, t_b, lat_b, lon_b = _extract_obs_info(obs_b)

    from .temporal import check_temporal_comparability

    cam_meta = road_graph.metadata if road_graph and hasattr(road_graph, "metadata") else None
    t_comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=cam_meta)

    if t_comp["comparable"]:
        delta_t = t_comp["delta_seconds"]
    elif t_comp["status"] == "invalid_negative_time":
        delta_t = -1.0
    else:
        delta_t = None

    node_a = road_graph.associate_camera(cam_a, latitude=lat_a, longitude=lon_a)
    node_b = road_graph.associate_camera(cam_b, latitude=lat_b, longitude=lon_b)

    start_obs_dict = {
        "observation_id": id_a,
        "camera_id": cam_a,
        "timestamp_seconds": t_a,
        "node_id": node_a,
    }
    end_obs_dict = {
        "observation_id": id_b,
        "camera_id": cam_b,
        "timestamp_seconds": t_b,
        "node_id": node_b,
    }

    gap_id = f"gap_{id_a}->{id_b}"

    # 1. Temporal Inversion / Non-positive Time Difference
    if t_comp["status"] == "invalid_negative_time" or (delta_t is not None and delta_t <= 0):
        if t_comp["status"] == "invalid_negative_time" or (delta_t is not None and delta_t < 0):
            status = "temporal_inversion"
            reason = f"Negative elapsed time: Observation timestamps are reversed ({t_comp.get('reason')})."
            gap_dur = None
        else:
            status = "infeasible"
            reason = f"Simultaneous observations at distinct locations with zero elapsed time (Δt = 0.0s)."
            gap_dur = 0.0

        return SparseObservationGap(
            gap_id=gap_id,
            identity_id=identity_id,
            start_observation=start_obs_dict,
            end_observation=end_obs_dict,
            gap_duration_seconds=gap_dur,
            start_node_id=node_a,
            end_node_id=node_b,
            candidate_routes=[],
            confidence=0.0,
            is_ambiguous=False,
            ambiguity_reason=reason,
            status=status,
            gap_state="invalid_time_interval",
            unobserved_intermediate_nodes=[],
            metadata={"delta_t_seconds": delta_t, "temporal_evidence": t_comp},
        )

    # 2. Camera Association Verification
    if node_a is None or node_b is None:
        unassoc = []
        if node_a is None:
            unassoc.append(f"start camera '{cam_a}'")
        if node_b is None:
            unassoc.append(f"destination camera '{cam_b}'")
        reason = f"Camera could not be associated with road network: {', '.join(unassoc)}."

        return SparseObservationGap(
            gap_id=gap_id,
            identity_id=identity_id,
            start_observation=start_obs_dict,
            end_observation=end_obs_dict,
            gap_duration_seconds=delta_t,
            start_node_id=node_a,
            end_node_id=node_b,
            candidate_routes=[],
            confidence=0.0,
            is_ambiguous=False,
            ambiguity_reason=reason,
            status="unassociated_camera",
            gap_state="unavailable_camera_evidence",
            unobserved_intermediate_nodes=[],
            metadata={"unassociated_cameras": unassoc},
        )

    # 3. Same Node (Stationary)
    if node_a == node_b:
        c_route = CandidateRoute(
            route_id="route_stationary",
            edges=[],
            nodes=[node_a],
            distance_meters=0.0,
            estimated_travel_time_seconds=0.0,
            min_travel_time_seconds=0.0,
            required_speed_kmh=0.0,
            speed_limit_kmh=50.0,
            feasible=True,
            feasibility_status="stationary",
            estimated_likelihood=1.0,
            raw_score=1.0,
            explanation=f"Stationary or loitering vehicle observed at same junction {node_a} across {delta_t:.1f}s interval.",
        )
        return SparseObservationGap(
            gap_id=gap_id,
            identity_id=identity_id,
            start_observation=start_obs_dict,
            end_observation=end_obs_dict,
            gap_duration_seconds=delta_t,
            start_node_id=node_a,
            end_node_id=node_b,
            candidate_routes=[c_route],
            confidence=1.0,
            is_ambiguous=False,
            ambiguity_reason=None,
            status="success",
            gap_state="observed_endpoints_with_unobserved_interval",
            unobserved_intermediate_nodes=[],
            metadata={"is_stationary": True},
        )

    # 4. Candidate Hidden Route Discovery via RoadGraph
    raw_paths = road_graph.find_candidate_paths(node_a, node_b, max_paths=max_paths)

    if not raw_paths:
        return SparseObservationGap(
            gap_id=gap_id,
            identity_id=identity_id,
            start_observation=start_obs_dict,
            end_observation=end_obs_dict,
            gap_duration_seconds=delta_t,
            start_node_id=node_a,
            end_node_id=node_b,
            candidate_routes=[],
            confidence=0.0,
            is_ambiguous=False,
            ambiguity_reason=f"No connected path exists between {node_a} and {node_b} respecting directed road restrictions and closures.",
            status="no_path",
            gap_state="no_feasible_route",
            unobserved_intermediate_nodes=[],
            metadata={"paths_explored": 0},
        )

    shortest_dist = min(p["distance_m"] for p in raw_paths)

    # 5. Evaluate Temporal Feasibility & Score Each Candidate Route
    evaluated_routes: List[CandidateRoute] = []
    feasible_routes: List[CandidateRoute] = []
    feasible_intermediate_nodes: Set[str] = set()

    for idx, path_data in enumerate(raw_paths):
        route_id = f"hidden_route_{idx + 1:02d}"
        edges = path_data["edges"]
        nodes = path_data["nodes"]
        dist_m = path_data["distance_m"]
        speed_lim = path_data.get("speed_limit_kmh", 50.0)

        # Road-aware feasibility evaluation
        is_feasible, status_code, req_speed, raw_score, est_travel_time = evaluate_route_feasibility_and_score(
            path_data=path_data,
            delta_t_seconds=delta_t,
            shortest_distance_m=shortest_dist,
            config=config,
        )

        min_travel_time = (dist_m / (speed_lim / 3.6)) if speed_lim > 0 else 0.0

        # Deterministic explainability string
        if is_feasible:
            if status_code == "feasible_temporally_unverified":
                explanation = (
                    f"Temporally unverified hidden route ({dist_m:.1f}m): Road connectivity and topology feasible, "
                    f"but cross-camera temporal evidence is unavailable ({t_comp.get('reason')})."
                )
            else:
                explanation = (
                    f"Feasible hidden route ({dist_m:.1f}m): Required speed {req_speed:.1f} km/h is compatible "
                    f"with road speed limit ({speed_lim:.1f} km/h) over {delta_t:.1f}s gap."
                )
            # Intermediate nodes of this feasible route
            for node in nodes[1:-1]:
                feasible_intermediate_nodes.add(node)
        else:
            if status_code == "speed_limit_exceeded":
                explanation = (
                    f"Infeasible hidden route ({dist_m:.1f}m): Required speed {req_speed:.1f} km/h exceeds "
                    f"effective speed limit tolerance ({speed_lim * 1.25:.1f} km/h) over {delta_t:.1f}s gap."
                )
            else:
                explanation = f"Infeasible hidden route: Temporal constraint violated ({status_code})."

        cand_route = CandidateRoute(
            route_id=route_id,
            edges=edges,
            nodes=nodes,
            distance_meters=dist_m,
            estimated_travel_time_seconds=est_travel_time,
            min_travel_time_seconds=min_travel_time,
            required_speed_kmh=req_speed,
            speed_limit_kmh=speed_lim,
            feasible=is_feasible,
            feasibility_status=status_code,
            estimated_likelihood=0.0,
            raw_score=raw_score if is_feasible else 0.0,
            explanation=explanation,
            temporal_evidence=t_comp,
        )

        evaluated_routes.append(cand_route)
        if is_feasible:
            feasible_routes.append(cand_route)

    # 6. Route Likelihood Normalization (Relative Estimated Likelihoods)
    total_raw_score = sum(r.raw_score for r in feasible_routes)
    if feasible_routes and total_raw_score > 0:
        for r in feasible_routes:
            r.estimated_likelihood = r.raw_score / total_raw_score
    elif feasible_routes:
        equal_prob = 1.0 / len(feasible_routes)
        for r in feasible_routes:
            r.estimated_likelihood = equal_prob

    # Sort descending: feasible first, then highest likelihood, then shortest distance
    evaluated_routes.sort(key=lambda r: (r.feasible, r.estimated_likelihood, -r.distance_meters), reverse=True)

    # 7. Uncertainty Preservation & Ambiguity Check
    is_ambiguous = False
    ambiguity_reason = None
    status = "success" if feasible_routes else "infeasible"
    confidence = evaluated_routes[0].estimated_likelihood if feasible_routes else 0.0
    unobserved_intermediate_list = sorted(list(feasible_intermediate_nodes))

    most_likely_intermediate_nodes = (
        list(feasible_routes[0].nodes[1:-1]) if feasible_routes and len(feasible_routes[0].nodes) > 2 else []
    )
    common_intermediate_nodes: List[str] = []
    if feasible_routes:
        common_set = set(feasible_routes[0].nodes[1:-1])
        for r in feasible_routes[1:]:
            common_set.intersection_update(r.nodes[1:-1])
        common_intermediate_nodes = sorted(list(common_set))

    if feasible_routes:
        if len(feasible_routes) > 1:
            diff = evaluated_routes[0].estimated_likelihood - evaluated_routes[1].estimated_likelihood
            if diff < ambiguity_threshold:
                is_ambiguous = True
                ambiguity_reason = (
                    f"Unobserved gap trajectory is ambiguous between Route 1 ({evaluated_routes[0].estimated_likelihood * 100:.1f}%) "
                    f"and Route 2 ({evaluated_routes[1].estimated_likelihood * 100:.1f}%)."
                )
        gap_state = "missing_intermediate_observations" if unobserved_intermediate_list else "observed_endpoints_with_unobserved_interval"
    else:
        gap_state = "no_feasible_route"
        if delta_t is not None:
            ambiguity_reason = f"All {len(evaluated_routes)} candidate routes between {node_a} and {node_b} are physically infeasible within {delta_t:.1f}s."
        else:
            ambiguity_reason = f"All {len(evaluated_routes)} candidate routes between {node_a} and {node_b} are infeasible."

    # Day 5: Propagate reliability and uncertainty into gap model
    from .reliability_engine import evaluate_observation_reliability, propagate_trajectory_uncertainty

    obs_rel_a = evaluate_observation_reliability(obs_a, config=config)
    obs_rel_b = evaluate_observation_reliability(obs_b, config=config)
    is_gap_interval = len(unobserved_intermediate_list) > 0 or gap_state == "missing_intermediate_observations"
    gap_rel = propagate_trajectory_uncertainty(
        trajectory_id=gap_id,
        obs_rel_a=obs_rel_a,
        obs_rel_b=obs_rel_b,
        candidate_routes=evaluated_routes,
        is_gap=is_gap_interval,
        unobserved_intermediate_nodes=unobserved_intermediate_list,
        gap_duration_seconds=delta_t,
        config=config,
    )

    return SparseObservationGap(
        gap_id=gap_id,
        identity_id=identity_id,
        start_observation=start_obs_dict,
        end_observation=end_obs_dict,
        gap_duration_seconds=delta_t,
        start_node_id=node_a,
        end_node_id=node_b,
        candidate_routes=evaluated_routes,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        ambiguity_reason=ambiguity_reason,
        status=status,
        gap_state=gap_state,
        unobserved_intermediate_nodes=unobserved_intermediate_list,
        metadata={
            "delta_t_seconds": delta_t,
            "temporal_evidence": t_comp,
            "candidate_routes_count": len(evaluated_routes),
            "feasible_routes_count": len(feasible_routes),
            "intermediate_node_semantics": "union_of_feasible_candidate_routes",
            "most_likely_route_intermediate_nodes": most_likely_intermediate_nodes,
            "common_intermediate_nodes": common_intermediate_nodes,
        },
        reliability=gap_rel.to_dict(),
        uncertainty=gap_rel.to_dict(),
    )


def infer_sparse_identity_trajectory(
    identity_data: Dict[str, Any],
    road_graph: RoadGraph,
    config: Optional[Dict[str, Any]] = None,
    max_paths: int = 5,
) -> VehicleTrajectory:
    """
    Reconstruct an entire multi-observation vehicle identity trajectory,
    explicitly detecting and inferring unobserved gaps between sightings.

    Chains sequential observations without synthesizing fake intermediate observations.
    For each interval, assesses whether it represents an unobserved gap,
    evaluates road-aware candidate hidden routes, and populates gap metadata.

    Args:
        identity_data: Identity dictionary with member observations.
        road_graph: RoadGraph spatial network instance.
        config: Optional configuration dictionary.
        max_paths: Maximum candidate paths per interval.

    Returns:
        VehicleTrajectory: Full trajectory hypothesis with gap metrics.
    """
    config = config or {}
    ident_id = identity_data.get("identity_id", identity_data.get("candidate_vehicle_id", "vehicle_candidate"))

    raw_obs_list = identity_data.get("member_observations") or identity_data.get("observations") or []
    sorted_obs = sorted(
        raw_obs_list,
        key=lambda o: _extract_obs_info(o)[2],
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
            gaps_count=0,
        )

    cameras = list(dict.fromkeys([str(o.get("camera_id", "")) for o in sorted_obs]))
    t_start = _extract_obs_info(sorted_obs[0])[2]
    t_end = _extract_obs_info(sorted_obs[-1])[2]
    total_time = t_end - t_start

    # Single observation (singleton / stationary)
    if n_obs == 1:
        _, cam_single, _, lat_s, lon_s = _extract_obs_info(sorted_obs[0])
        node_s = road_graph.associate_camera(cam_single, latitude=lat_s, longitude=lon_s)
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
            complete_route_nodes=[node_s] if node_s else [],
            gaps_count=0,
        )

    segments: List[TrajectorySegment] = []
    complete_edges: List[str] = []
    complete_nodes: List[str] = []
    total_dist = 0.0
    conf_scores: List[float] = []
    any_ambiguous = False
    gaps_count = 0

    for i in range(n_obs - 1):
        obs_curr = sorted_obs[i]
        obs_next = sorted_obs[i + 1]

        # Use infer_sparse_gap for full gap model evaluation
        gap = infer_sparse_gap(
            obs_curr,
            obs_next,
            road_graph,
            identity_id=ident_id,
            max_paths=max_paths,
            config=config,
        )

        # Check whether this interval is an unobserved gap
        is_gap = len(gap.unobserved_intermediate_nodes) > 0 or gap.gap_state == "missing_intermediate_observations"
        if is_gap:
            gaps_count += 1

        seg = TrajectorySegment(
            segment_id=gap.gap_id,
            identity_id=ident_id,
            start_observation_id=gap.start_observation["observation_id"],
            start_camera_id=gap.start_observation["camera_id"],
            start_timestamp=gap.start_observation["timestamp_seconds"],
            end_observation_id=gap.end_observation["observation_id"],
            end_camera_id=gap.end_observation["camera_id"],
            end_timestamp=gap.end_observation["timestamp_seconds"],
            time_difference_seconds=gap.gap_duration_seconds,
            start_node_id=gap.start_node_id,
            end_node_id=gap.end_node_id,
            candidate_routes=gap.candidate_routes,
            most_likely_route=gap.most_likely_route,
            confidence=gap.confidence,
            is_ambiguous=gap.is_ambiguous,
            ambiguity_reason=gap.ambiguity_reason,
            status=gap.status,
            is_gap=is_gap,
            gap_duration_seconds=gap.gap_duration_seconds if is_gap else None,
            gap_state=gap.gap_state,
            reliability=gap.reliability,
            uncertainty=gap.uncertainty,
        )
        segments.append(seg)

        if seg.is_ambiguous:
            any_ambiguous = True

        if seg.most_likely_route is not None:
            complete_edges.extend(seg.most_likely_route)
            if seg.candidate_routes and seg.candidate_routes[0].nodes:
                seg_nodes = seg.candidate_routes[0].nodes
                if not complete_nodes:
                    complete_nodes.extend(seg_nodes)
                else:
                    complete_nodes.extend(seg_nodes[1:])
            total_dist += (seg.candidate_routes[0].distance_meters if seg.candidate_routes else 0.0)
            conf_scores.append(seg.confidence)
        else:
            conf_scores.append(0.0)

    # Compute overall confidence
    if conf_scores and all(c > 0 for c in conf_scores):
        log_sum = sum(math.log(c) for c in conf_scores)
        overall_conf = math.exp(log_sum / len(conf_scores))
    elif conf_scores:
        overall_conf = sum(conf_scores) / len(conf_scores)
    else:
        overall_conf = 0.0

    if segments:
        avg_rel = sum((s.reliability or {}).get("overall_reliability", 0.80) for s in segments) / len(segments)
        avg_unc = round(1.0 - avg_rel, 4)
        traj_rel = {"overall_reliability": round(avg_rel, 4), "overall_uncertainty": avg_unc, "segments_count": len(segments)}
        traj_unc = {"score": avg_unc, "level": "low" if avg_unc <= 0.25 else ("moderate" if avg_unc <= 0.50 else ("high" if avg_unc <= 0.75 else "critical"))}
    else:
        traj_rel = {"overall_reliability": 1.0, "overall_uncertainty": 0.0}
        traj_unc = {"score": 0.0, "level": "low"}

    has_unavailable_time = any(s.time_difference_seconds is None for s in segments)
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
        complete_route_edges=complete_edges,
        complete_route_nodes=complete_nodes,
        gaps_count=gaps_count,
        reliability=traj_rel,
        uncertainty=traj_unc,
    )

