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
from .similarity import appearance_similarity, validate_and_normalize_embedding


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

    # Metadata & Member Observations
    member_observations: List[Observation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "vehicle_type": self.vehicle_type,
            "observations_count": self.observations_count,
            "start_timestamp_seconds": round(self.start_timestamp_seconds, 2),
            "end_timestamp_seconds": round(self.end_timestamp_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "frame_ids": list(self.frame_ids),
            "aggregated_plate": self.aggregated_plate,
            "aggregated_plate_confidence": (
                round(self.aggregated_plate_confidence, 4)
                if self.aggregated_plate_confidence is not None
                else None
            ),
            "plate_votes_count": self.plate_votes_count,
            "aggregated_embedding_dim": (
                len(self.aggregated_embedding)
                if self.aggregated_embedding is not None
                else 0
            ),
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
        first_obs = self.member_observations[0] if self.member_observations else None

        return Observation(
            observation_id=f"track_{self.camera_id}_{self.track_id}",
            camera_id=self.camera_id,
            timestamp_seconds=chosen_ts,
            vehicle_type=self.vehicle_type,
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


def pool_embeddings(observations: List[Observation]) -> Optional[List[float]]:
    """
    Calculate confidence-weighted centroid of appearance embeddings across frames,
    then apply unit L2-normalization.
    """
    valid_embs = []
    weights = []

    for obs in observations:
        if obs.appearance_embedding and isinstance(obs.appearance_embedding, (list, tuple)):
            norm_emb = validate_and_normalize_embedding(obs.appearance_embedding)
            if norm_emb is not None:
                w = float(obs.detection_confidence if obs.detection_confidence is not None else 1.0)
                valid_embs.append(norm_emb)
                weights.append(max(0.1, w))

    if not valid_embs:
        return None

    dim = len(valid_embs[0])
    # Verify all embeddings have identical dimensionality
    valid_embs_clean = [e for e in valid_embs if len(e) == dim]
    if not valid_embs_clean:
        return None

    total_weight = sum(weights[:len(valid_embs_clean)])
    pooled = [0.0] * dim

    for emb, w in zip(valid_embs_clean, weights[:len(valid_embs_clean)]):
        for i in range(dim):
            pooled[i] += emb[i] * w

    if total_weight > 0:
        pooled = [x / total_weight for x in pooled]

    return validate_and_normalize_embedding(pooled)


def aggregate_observations_into_tracklets(
    observations: List[Observation],
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
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

        # Vehicle type: majority vote
        types = [o.vehicle_type for o in sorted_obs if o.vehicle_type]
        v_type = Counter(types).most_common(1)[0][0] if types else None

        # Plate aggregation
        agg_plate, agg_plate_conf, plate_votes = aggregate_plate_votes(sorted_obs)

        # Embedding pooling
        pooled_emb = pool_embeddings(sorted_obs)

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
