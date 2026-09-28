"""
Track-Level Reasoning Engine for UrbanTrack AI.
Aggregates camera-local multi-frame vehicle detections into consolidated Tracklets
before performing cross-camera identity fusion.

Solves Phase 11:
- Multi-frame OCR consensus voting weighted by confidence
- Quality-weighted appearance feature pooling and L2 normalization
- Temporal duration and bounding timestamps [t_start, t_end]
- Camera reliability and detection confidence propagation
- Image-space motion consistency
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from schemas.observation_schema import Observation
from .identity_fusion import match_observations
from .similarity import (
    HARD_INCOMPATIBLE_VEHICLE_TYPES,
    appearance_similarity,
    are_reid_models_compatible,
    validate_and_normalize_embedding,
)


@dataclass
class Tracklet:
    """
    Consolidated representation of a camera-local vehicle tracklet.
    Aggregates multi-frame observations belonging to the same camera and track_id.
    """
    track_id: str
    camera_id: str
    vehicle_type: Optional[str]
    observations_count: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    duration_seconds: float
    frame_ids: List[int]

    # Aggregated Identity Evidence
    aggregated_plate: Optional[str]
    aggregated_plate_confidence: Optional[float]
    plate_votes_count: int
    aggregated_embedding: Optional[List[float]]

    # Quality & Reliability
    average_detection_confidence: Optional[float]
    camera_reliability: Optional[float]

    # Kinematics & Coordinates
    latitude: Optional[float]
    longitude: Optional[float]
    average_pixel_speed: Optional[float]
    heading_angle: Optional[float]

    # Phase 8: Enhanced Tracklet Representation
    synchronized_start_timestamp_seconds: Optional[float] = None
    synchronized_end_timestamp_seconds: Optional[float] = None
    embedding_dispersion: Optional[float] = None
    embedding_model: Optional[str] = None
    ocr_confidence_stats: Optional[Dict[str, float]] = None
    vehicle_type_consensus: Optional[str] = None
    world_coordinate_quality: Optional[str] = None
    observation_quality: Optional[float] = None

    # Metadata & Member Observations
    member_observations: List[Observation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "vehicle_type": self.vehicle_type,
            "vehicle_type_consensus": self.vehicle_type_consensus or self.vehicle_type,
            "observations_count": self.observations_count,
            "start_timestamp_seconds": round(self.start_timestamp_seconds, 2),
            "end_timestamp_seconds": round(self.end_timestamp_seconds, 2),
            "synchronized_start_timestamp_seconds": (
                round(self.synchronized_start_timestamp_seconds, 2)
                if self.synchronized_start_timestamp_seconds is not None
                else None
            ),
            "synchronized_end_timestamp_seconds": (
                round(self.synchronized_end_timestamp_seconds, 2)
                if self.synchronized_end_timestamp_seconds is not None
                else None
            ),
            "duration_seconds": round(self.duration_seconds, 2),
            "frame_ids": list(self.frame_ids),
            "aggregated_plate": self.aggregated_plate,
            "aggregated_plate_confidence": (
                round(self.aggregated_plate_confidence, 4)
                if self.aggregated_plate_confidence is not None
                else None
            ),
            "plate_votes_count": self.plate_votes_count,
            "ocr_confidence_stats": self.ocr_confidence_stats,
            "aggregated_embedding_dim": (
                len(self.aggregated_embedding)
                if self.aggregated_embedding is not None
                else 0
            ),
            "embedding_dispersion": self.embedding_dispersion,
            "embedding_model": self.embedding_model,
            "average_detection_confidence": (
                round(self.average_detection_confidence, 4)
                if self.average_detection_confidence is not None
                else None
            ),
            "camera_reliability": (
                round(self.camera_reliability, 4)
                if self.camera_reliability is not None
                else None
            ),
            "world_coordinate_quality": self.world_coordinate_quality,
            "observation_quality": self.observation_quality,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "average_pixel_speed": (
                round(self.average_pixel_speed, 2)
                if self.average_pixel_speed is not None
                else None
            ),
            "heading_angle": (
                round(self.heading_angle, 2)
                if self.heading_angle is not None
                else None
            ),
        }

    def to_representative_observation(self, use_exit: bool = True) -> Observation:
        """
        Create a representative Observation representing this tracklet.
        If use_exit is True, uses end timestamp (vehicle leaving camera FOV).
        """
        chosen_ts = self.end_timestamp_seconds if use_exit else self.start_timestamp_seconds
        chosen_sync_ts = (
            self.synchronized_end_timestamp_seconds if use_exit else self.synchronized_start_timestamp_seconds
        )
        first_obs = self.member_observations[0] if self.member_observations else None
        last_obs = self.member_observations[-1] if self.member_observations else None
        target_obs = last_obs if use_exit and last_obs else first_obs
        canonical_id = first_obs.observation_id if first_obs else f"track_{self.camera_id}_{self.track_id}"

        obs = Observation(
            observation_id=canonical_id,
            camera_id=self.camera_id,
            timestamp_seconds=chosen_ts,
            vehicle_type=self.vehicle_type_consensus or self.vehicle_type,
            plate=self.aggregated_plate,
            plate_confidence=self.aggregated_plate_confidence,
            appearance_embedding=self.aggregated_embedding,
            camera_reliability=self.camera_reliability,
            latitude=self.latitude,
            longitude=self.longitude,
            track_id=self.track_id,
            detection_confidence=self.average_detection_confidence,
            timestamp_semantics=first_obs.timestamp_semantics if first_obs else "synchronized",
            time_reference_id=first_obs.time_reference_id if first_obs else "city_sync_grid",
        )
        if self.member_observations:
            setattr(obs, "member_observations", self.member_observations)
        if self.embedding_model:
            setattr(obs, "embedding_model", self.embedding_model)
            setattr(obs, "reid_model", self.embedding_model)
        if chosen_sync_ts is not None:
            setattr(obs, "synchronized_timestamp_seconds", chosen_sync_ts)
        if target_obs and getattr(target_obs, "world_position", None) is not None:
            setattr(obs, "world_position", target_obs.world_position)
            setattr(obs, "world_coordinate_system", getattr(target_obs, "world_coordinate_system", None))
        return obs


def aggregate_plate_votes(observations: List[Observation]) -> Tuple[Optional[str], Optional[float], int]:
    """
    Perform confidence-weighted character frequency voting across multiple frame OCR detections.
    Filters out transient OCR jitter and misread characters.
    """
    valid_plates = []
    for obs in observations:
        p = obs.plate or obs.plate_text
        if p:
            clean = "".join(c for c in str(p).upper() if c.isalnum())
            if len(clean) >= 3:
                conf = float(obs.plate_confidence if obs.plate_confidence is not None else (obs.ocr_confidence if obs.ocr_confidence is not None else 0.80))
                valid_plates.append((clean, conf))

    if not valid_plates:
        return None, None, 0

    # 1. Exact plate string frequency weighted by confidence
    string_scores: Dict[str, float] = {}
    for p_str, conf in valid_plates:
        string_scores[p_str] = string_scores.get(p_str, 0.0) + conf

    # 2. Length consensus
    length_counts = Counter(len(p) for p, _ in valid_plates)
    consensus_len = length_counts.most_common(1)[0][0]

    # 3. Position-wise character voting among strings of consensus length
    candidate_strings = [p for p, _ in valid_plates if len(p) == consensus_len]
    if candidate_strings:
        consensus_chars = []
        for pos in range(consensus_len):
            char_weights: Dict[str, float] = {}
            for p, conf in valid_plates:
                if len(p) == consensus_len:
                    ch = p[pos]
                    char_weights[ch] = char_weights.get(ch, 0.0) + conf
            best_ch = max(char_weights.items(), key=lambda item: item[1])[0]
            consensus_chars.append(best_ch)
        consensus_plate = "".join(consensus_chars)
    else:
        consensus_plate = max(string_scores.items(), key=lambda item: item[1])[0]

    # Calculate average confidence for the consensus plate
    matching_confs = [conf for p, conf in valid_plates if p == consensus_plate]
    avg_conf = (sum(matching_confs) / len(matching_confs)) if matching_confs else (sum(c for _, c in valid_plates) / len(valid_plates))

    return consensus_plate, round(avg_conf, 4), len(valid_plates)


def compute_embedding_dispersion(embeddings: List[List[float]]) -> Optional[float]:
    """
    Compute intra-tracklet embedding dispersion as mean pairwise cosine distance:
    dispersion = 1.0 - mean(cosine_similarity(e_i, e_j)) for i < j.
    Returns 0.0 for single-observation tracklets, or None if no embeddings.
    """
    if not embeddings or len(embeddings) < 2:
        return 0.0 if embeddings else None

    sims = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            sim = appearance_similarity(embeddings[i], embeddings[j])
            if sim is not None:
                sims.append(sim)
    if not sims:
        return 0.0
    mean_sim = sum(sims) / len(sims)
    return round(max(0.0, 1.0 - mean_sim), 4)


def pool_embeddings(
    observations: List[Observation],
    strategy: str = "weighted_mean",
) -> Optional[List[float]]:
    """
    Consolidate appearance embeddings across tracklet observations using specified pooling strategy.

    Strategies:
        - 'weighted_mean': Detection-confidence weighted centroid, L2-normalized (default).
        - 'mean': Unweighted centroid, L2-normalized.
        - 'medoid': The individual frame embedding with minimum average cosine distance to all other frames.
        - 'best_quality': The embedding from the frame with highest detection confidence.

    Returns:
        Optional[List[float]]: Representative unit L2-normalized embedding.
    """
    valid_items: List[Tuple[List[float], float]] = []

    for obs in observations:
        if obs.appearance_embedding and isinstance(obs.appearance_embedding, (list, tuple)):
            norm_emb = validate_and_normalize_embedding(obs.appearance_embedding)
            if norm_emb is not None:
                w = float(obs.detection_confidence if obs.detection_confidence is not None else 1.0)
                valid_items.append((norm_emb, max(0.1, w)))

    if not valid_items:
        return None

    dim = len(valid_items[0][0])
    valid_items = [item for item in valid_items if len(item[0]) == dim]
    if not valid_items:
        return None

    if len(valid_items) == 1 or strategy == "best_quality":
        best = max(valid_items, key=lambda it: it[1])
        return best[0]

    if strategy == "medoid":
        embs = [it[0] for it in valid_items]
        n = len(embs)
        best_idx = 0
        best_avg_sim = -1.0
        for i in range(n):
            sims = [appearance_similarity(embs[i], embs[j]) or 0.0 for j in range(n) if i != j]
            avg_sim = sum(sims) / len(sims) if sims else 1.0
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_idx = i
        return embs[best_idx]

    if strategy == "mean":
        pooled = [0.0] * dim
        for emb, _ in valid_items:
            for i in range(dim):
                pooled[i] += emb[i]
        pooled = [x / len(valid_items) for x in pooled]
        return validate_and_normalize_embedding(pooled)

    # Default: 'weighted_mean'
    total_weight = sum(w for _, w in valid_items)
    pooled = [0.0] * dim
    for emb, w in valid_items:
        for i in range(dim):
            pooled[i] += emb[i] * w

    if total_weight > 0:
        pooled = [x / total_weight for x in pooled]

    return validate_and_normalize_embedding(pooled)


def aggregate_observations_into_tracklets(
    observations: List[Observation],
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    embedding_pooling_strategy: str = "weighted_mean",
) -> List[Tracklet]:
    """
    Group frame-level observations by (camera_id, track_id) and build consolidated Tracklets.
    Falls back gracefully to individual singleton tracklets if track_id is absent.
    """
    camera_metadata = camera_metadata or {}
    groups: Dict[Tuple[str, str], List[Observation]] = {}

    for idx, obs in enumerate(observations):
        cam_id = obs.camera_id
        # Fallback to unique index if track_id is missing
        trk_id = str(obs.track_id) if obs.track_id is not None else f"untracked_{cam_id}_{idx:05d}"
        key = (cam_id, trk_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(obs)

    tracklets: List[Tracklet] = []

    for (cam_id, trk_id), obs_list in groups.items():
        # Sort chronologically by timestamp_seconds then frame_id
        sorted_obs = sorted(
            obs_list,
            key=lambda o: (o.timestamp_seconds, o.frame_id if o.frame_id is not None else 0),
        )

        n = len(sorted_obs)
        t_start = sorted_obs[0].timestamp_seconds
        t_end = sorted_obs[-1].timestamp_seconds
        duration = max(0.0, t_end - t_start)
        frames = [o.frame_id for o in sorted_obs if o.frame_id is not None]

        # Synchronized timestamps if available
        sync_ts_list = [
            getattr(o, "synchronized_timestamp_seconds", None)
            for o in sorted_obs
            if getattr(o, "synchronized_timestamp_seconds", None) is not None
        ]
        sync_start = sync_ts_list[0] if sync_ts_list else None
        sync_end = sync_ts_list[-1] if sync_ts_list else None

        # Vehicle type: majority vote
        types = [o.vehicle_type for o in sorted_obs if o.vehicle_type]
        v_type = Counter(types).most_common(1)[0][0] if types else None

        # Plate aggregation
        agg_plate, agg_plate_conf, plate_votes = aggregate_plate_votes(sorted_obs)

        # OCR stats
        plate_confs = [
            float(o.plate_confidence if o.plate_confidence is not None else (o.ocr_confidence if getattr(o, "ocr_confidence", None) is not None else 0.80))
            for o in sorted_obs if o.plate or getattr(o, "plate_text", None)
        ]
        ocr_stats = {
            "min": round(min(plate_confs), 4),
            "max": round(max(plate_confs), 4),
            "mean": round(sum(plate_confs) / len(plate_confs), 4),
        } if plate_confs else None

        # Embedding pooling with specified strategy
        pooled_emb = pool_embeddings(sorted_obs, strategy=embedding_pooling_strategy)

        # Embedding dispersion
        raw_embs = [
            validate_and_normalize_embedding(o.appearance_embedding)
            for o in sorted_obs if o.appearance_embedding
        ]
        raw_embs_clean = [e for e in raw_embs if e is not None]
        dispersion = compute_embedding_dispersion(raw_embs_clean)

        # Embedding model provenance
        models = [
            getattr(o, "embedding_model", None) or getattr(o, "reid_model", None)
            for o in sorted_obs
            if (getattr(o, "embedding_model", None) or getattr(o, "reid_model", None))
        ]
        emb_model = Counter(models).most_common(1)[0][0] if models else None

        # Average detection confidence
        det_confs = [o.detection_confidence for o in sorted_obs if o.detection_confidence is not None]
        avg_det_conf = (sum(det_confs) / len(det_confs)) if det_confs else None

        # Camera reliability from observations or metadata
        cam_rels = [o.camera_reliability for o in sorted_obs if o.camera_reliability is not None]
        if cam_rels:
            cam_rel = sum(cam_rels) / len(cam_rels)
        elif cam_id in camera_metadata and "reliability" in camera_metadata[cam_id]:
            cam_rel = float(camera_metadata[cam_id]["reliability"])
        else:
            cam_rel = None

        # Coordinates from observations or camera metadata
        lat = sorted_obs[0].latitude
        lon = sorted_obs[0].longitude
        if (lat is None or lon is None) and cam_id in camera_metadata:
            lat = camera_metadata[cam_id].get("latitude")
            lon = camera_metadata[cam_id].get("longitude")

        # World coordinate quality
        has_world = any(getattr(o, "world_position", None) is not None for o in sorted_obs)
        coord_quality = "calibrated_world_plane" if has_world else ("gps" if lat is not None else "image_space_only")

        # Kinematics
        speeds = [o.pixel_speed for o in sorted_obs if o.pixel_speed is not None]
        avg_speed = (sum(speeds) / len(speeds)) if speeds else None

        angles = [o.heading_angle for o in sorted_obs if o.heading_angle is not None]
        avg_angle = (sum(angles) / len(angles)) if angles else None

        trk = Tracklet(
            track_id=trk_id,
            camera_id=cam_id,
            vehicle_type=v_type,
            observations_count=n,
            start_timestamp_seconds=t_start,
            end_timestamp_seconds=t_end,
            duration_seconds=duration,
            frame_ids=frames,
            aggregated_plate=agg_plate,
            aggregated_plate_confidence=agg_plate_conf,
            plate_votes_count=plate_votes,
            aggregated_embedding=pooled_emb,
            average_detection_confidence=avg_det_conf,
            camera_reliability=cam_rel,
            latitude=lat,
            longitude=lon,
            average_pixel_speed=avg_speed,
            heading_angle=avg_angle,
            synchronized_start_timestamp_seconds=sync_start,
            synchronized_end_timestamp_seconds=sync_end,
            embedding_dispersion=dispersion,
            embedding_model=emb_model,
            ocr_confidence_stats=ocr_stats,
            vehicle_type_consensus=v_type,
            world_coordinate_quality=coord_quality,
            observation_quality=avg_det_conf,
            member_observations=sorted_obs,
        )
        tracklets.append(trk)

    # Sort tracklets deterministically: start_timestamp, camera_id, track_id
    tracklets.sort(key=lambda t: (t.start_timestamp_seconds, t.camera_id, t.track_id))
    return tracklets


def match_tracklets(
    tracklet_a: Tracklet,
    tracklet_b: Tracklet,
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform cross-camera identity fusion between two consolidated Tracklets.
    Uses tracklet_a's exit observation and tracklet_b's entry observation.
    """
    if tracklet_a.start_timestamp_seconds > tracklet_b.start_timestamp_seconds:
        first, second = tracklet_b, tracklet_a
    else:
        first, second = tracklet_a, tracklet_b

    # Exit of first tracklet -> Entry of second tracklet
    obs_exit = first.to_representative_observation(use_exit=True)
    obs_entry = second.to_representative_observation(use_exit=False)

    match_result = match_observations(
        obs_exit,
        obs_entry,
        camera_metadata=camera_metadata,
        config=config,
    )

    # Attach tracklet-specific evidence provenance
    match_result["tracklet_evidence"] = {
        "tracklet_a": {
            "track_id": first.track_id,
            "camera_id": first.camera_id,
            "observations_count": first.observations_count,
            "duration_seconds": first.duration_seconds,
            "plate_votes": first.plate_votes_count,
        },
        "tracklet_b": {
            "track_id": second.track_id,
            "camera_id": second.camera_id,
            "observations_count": second.observations_count,
            "duration_seconds": second.duration_seconds,
            "plate_votes": second.plate_votes_count,
        },
    }

    return match_result


@dataclass
class TrackletAssociationResult:
    """Result of bipartite tracklet-level association between camera pairs or window."""
    matched_pairs: List[Tuple[Tracklet, Tracklet, float, Dict[str, Any]]]
    unassigned_tracklets: List[Tracklet]
    algorithm: str
    execution_time_ms: float = 0.0
    pairwise_evaluations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "matched_pairs_count": len(self.matched_pairs),
            "unassigned_count": len(self.unassigned_tracklets),
            "execution_time_ms": round(self.execution_time_ms, 3),
            "pairwise_evaluations": self.pairwise_evaluations,
            "matched_pairs": [
                {
                    "source_camera": p[0].camera_id,
                    "source_track_id": p[0].track_id,
                    "target_camera": p[1].camera_id,
                    "target_track_id": p[1].track_id,
                    "score": round(p[2], 4),
                    "decision_state": p[3].get("decision_state", "UNKNOWN"),
                }
                for p in self.matched_pairs
            ],
        }


class TrackletAssociator:
    """
    Sliding-window bipartite tracklet association engine.
    Solves 1-to-1 global vehicle identity assignment across camera transitions
    using Hungarian (linear_sum_assignment) or Greedy matching.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.70,
        max_time_window_seconds: float = 600.0,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.max_time_window_seconds = max_time_window_seconds
        self.camera_metadata = camera_metadata or {}
        self.config = config or {}

    def associate_tracklets(
        self,
        source_tracklets: List[Tracklet],
        target_tracklets: List[Tracklet],
        method: str = "hungarian",
    ) -> TrackletAssociationResult:
        """
        Execute 1-to-1 bipartite assignment from source to target tracklets.
        """
        import time
        t0 = time.perf_counter()

        if not source_tracklets or not target_tracklets:
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            unassigned = list(source_tracklets) + list(target_tracklets)
            return TrackletAssociationResult(
                matched_pairs=[],
                unassigned_tracklets=unassigned,
                algorithm=method,
                execution_time_ms=t_elapsed,
                pairwise_evaluations=0,
            )

        n_src = len(source_tracklets)
        n_tgt = len(target_tracklets)
        affinity_matrix = [[0.0 for _ in range(n_tgt)] for _ in range(n_src)]
        match_records: Dict[Tuple[int, int], Dict[str, Any]] = {}
        eval_count = 0

        for i, src in enumerate(source_tracklets):
            t_src_exit = (
                src.synchronized_end_timestamp_seconds
                if src.synchronized_end_timestamp_seconds is not None
                else src.end_timestamp_seconds
            )
            v_type_src = src.vehicle_type_consensus or src.vehicle_type

            for j, tgt in enumerate(target_tracklets):
                t_tgt_entry = (
                    tgt.synchronized_start_timestamp_seconds
                    if tgt.synchronized_start_timestamp_seconds is not None
                    else tgt.start_timestamp_seconds
                )
                dt = t_tgt_entry - t_src_exit
                if dt < -2.0 or dt > self.max_time_window_seconds:
                    continue

                # Hard vehicle type contradiction check
                v_type_tgt = tgt.vehicle_type_consensus or tgt.vehicle_type
                if v_type_src and v_type_tgt and (v_type_src, v_type_tgt) in HARD_INCOMPATIBLE_VEHICLE_TYPES:
                    continue

                # Re-ID model compatibility check: block cross-model comparison when no plates
                if src.embedding_model and tgt.embedding_model and not are_reid_models_compatible(src.embedding_model, tgt.embedding_model):
                    if not (src.aggregated_plate and tgt.aggregated_plate):
                        continue

                # Appearance feasibility quick check: skip pairs whose appearance is far below threshold when no plates exist
                if src.aggregated_embedding is not None and tgt.aggregated_embedding is not None:
                    sim = appearance_similarity(src.aggregated_embedding, tgt.aggregated_embedding)
                    if sim is not None and sim < (self.min_score_threshold - 0.15) and not (src.aggregated_plate and tgt.aggregated_plate):
                        continue

                eval_count += 1
                res = match_tracklets(src, tgt, camera_metadata=self.camera_metadata, config=self.config)
                score = float(res.get("same_vehicle_score", 0.0))
                affinity_matrix[i][j] = score
                match_records[(i, j)] = res

        matched_pairs: List[Tuple[Tracklet, Tracklet, float, Dict[str, Any]]] = []
        assigned_src: set[int] = set()
        assigned_tgt: set[int] = set()

        if method == "hungarian":
            from scipy.optimize import linear_sum_assignment
            cost_matrix = [[1.0 - affinity_matrix[i][j] for j in range(n_tgt)] for i in range(n_src)]
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                score = affinity_matrix[r][c]
                if score >= self.min_score_threshold:
                    rec = match_records.get((r, c), {})
                    matched_pairs.append((source_tracklets[r], target_tracklets[c], score, rec))
                    assigned_src.add(r)
                    assigned_tgt.add(c)
        elif method == "greedy":
            all_pairs = []
            for i in range(n_src):
                for j in range(n_tgt):
                    sc = affinity_matrix[i][j]
                    if sc >= self.min_score_threshold:
                        all_pairs.append((sc, i, j))
            all_pairs.sort(key=lambda item: -item[0])
            for sc, i, j in all_pairs:
                if i not in assigned_src and j not in assigned_tgt:
                    rec = match_records.get((i, j), {})
                    matched_pairs.append((source_tracklets[i], target_tracklets[j], sc, rec))
                    assigned_src.add(i)
                    assigned_tgt.add(j)
        else:
            raise ValueError(f"Unknown association method: {method}. Must be 'hungarian' or 'greedy'.")

        unassigned = [source_tracklets[i] for i in range(n_src) if i not in assigned_src] + \
                     [target_tracklets[j] for j in range(n_tgt) if j not in assigned_tgt]

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        return TrackletAssociationResult(
            matched_pairs=matched_pairs,
            unassigned_tracklets=unassigned,
            algorithm=method,
            execution_time_ms=t_elapsed,
            pairwise_evaluations=eval_count,
        )

    def benchmark_assignment_methods(
        self,
        source_tracklets: List[Tracklet],
        target_tracklets: List[Tracklet],
    ) -> Dict[str, Any]:
        """Benchmark Hungarian vs Greedy assignment on identical tracklet pairs."""
        res_hungarian = self.associate_tracklets(source_tracklets, target_tracklets, method="hungarian")
        res_greedy = self.associate_tracklets(source_tracklets, target_tracklets, method="greedy")

        h_scores = [p[2] for p in res_hungarian.matched_pairs]
        g_scores = [p[2] for p in res_greedy.matched_pairs]

        return {
            "hungarian": {
                "matches": len(res_hungarian.matched_pairs),
                "mean_score": round(sum(h_scores) / len(h_scores), 4) if h_scores else 0.0,
                "execution_time_ms": res_hungarian.execution_time_ms,
            },
            "greedy": {
                "matches": len(res_greedy.matched_pairs),
                "mean_score": round(sum(g_scores) / len(g_scores), 4) if g_scores else 0.0,
                "execution_time_ms": res_greedy.execution_time_ms,
            },
            "speedup_ratio": (
                round(res_hungarian.execution_time_ms / res_greedy.execution_time_ms, 2)
                if res_greedy.execution_time_ms > 0
                else 1.0
            ),
        }

    def associate_multicamera_network(
        self,
        tracklets: List[Tracklet],
        method: str = "hungarian",
    ) -> TrackletAssociationResult:
        """
        Execute global tracklet association across an entire multi-camera sensor network.
        Evaluates candidate transitions across all camera pairs, applies bipartite 1-to-1 matching
        per transition with hierarchical evidence evaluation, and returns all matched pairs.
        """
        import time
        t0 = time.perf_counter()

        if not tracklets:
            return TrackletAssociationResult(
                matched_pairs=[],
                unassigned_tracklets=[],
                algorithm=method,
                execution_time_ms=0.0,
                pairwise_evaluations=0,
            )

        # Partition tracklets by camera
        cam_groups: Dict[str, List[Tracklet]] = {}
        for trk in tracklets:
            cam_groups.setdefault(trk.camera_id, []).append(trk)

        cameras = sorted(list(cam_groups.keys()))
        all_matched_pairs: List[Tuple[Tracklet, Tracklet, float, Dict[str, Any]]] = []
        total_evaluations = 0

        # Evaluate transitions between every distinct camera pair (chronologically oriented)
        for i in range(len(cameras)):
            for j in range(len(cameras)):
                if i == j:
                    continue
                cam_a = cameras[i]
                cam_b = cameras[j]
                src_list = cam_groups[cam_a]
                tgt_list = cam_groups[cam_b]

                # Filter target tracklets temporally plausible relative to source tracklets
                res = self.associate_tracklets(src_list, tgt_list, method=method)
                total_evaluations += res.pairwise_evaluations
                all_matched_pairs.extend(res.matched_pairs)

        matched_tracklet_ids = set()
        for p in all_matched_pairs:
            matched_tracklet_ids.add((p[0].camera_id, p[0].track_id))
            matched_tracklet_ids.add((p[1].camera_id, p[1].track_id))

        unassigned = [
            trk for trk in tracklets
            if (trk.camera_id, trk.track_id) not in matched_tracklet_ids
        ]

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        return TrackletAssociationResult(
            matched_pairs=all_matched_pairs,
            unassigned_tracklets=unassigned,
            algorithm=method,
            execution_time_ms=t_elapsed,
            pairwise_evaluations=total_evaluations,
        )
