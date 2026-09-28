"""
UrbanTrack AI — Official AI City 2022 Ground-Truth Adapter & Tracklet Linker.

Provides strict separation of ground-truth identities from the inference pipeline:
- Ground-truth vehicle IDs are NEVER injected into Observation inference fields.
- Ground truth is stored in an independent evaluation-only registry.
- Implements spatiotemporal consensus IoU linking between Member-1 tracklets and GT boxes.
- Generates pairwise ground-truth relationships (SAME_VEHICLE / DIFFERENT_VEHICLE).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from schemas.observation_schema import Observation


def compute_iou(box1: List[float] | Tuple[float, float, float, float], box2: List[float] | Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


@dataclass
class GroundTruthDetection:
    camera_id: str
    frame_id: int          # 0-indexed frame number to match perception
    gt_vehicle_id: int     # Official global vehicle ID
    bbox: List[float]      # [x1, y1, x2, y2]


@dataclass
class TrackletGTAssociation:
    observation_id: str
    camera_id: str
    local_track_id: int
    gt_vehicle_id: Optional[int]
    match_status: str      # 'matched', 'unmatched', 'ambiguous'
    frames_evaluated: int
    overlapping_frames: int
    consensus_ratio: float
    mean_iou: float
    competing_ids: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AICityGroundTruthAdapter:
    """
    Dedicated, isolated ground-truth adapter for AI City 2022 Track 1.
    """

    def __init__(
        self,
        gt_dir: Union[str, Path] = "data/aicity_ground_truth/gt",
        iou_threshold: float = 0.40,
        consensus_threshold: float = 0.60,
    ) -> None:
        self.gt_dir = Path(gt_dir)
        self.iou_threshold = iou_threshold
        self.consensus_threshold = consensus_threshold

        # Storage: camera_id -> frame_id -> list of GroundTruthDetection
        self.gt_by_frame: Dict[str, Dict[int, List[GroundTruthDetection]]] = {}
        self.gt_vehicle_ids: Set[int] = set()
        self.gt_vehicles_by_camera: Dict[str, Set[int]] = {}
        self.file_hashes: Dict[str, str] = {}

        # Mappings computed after linking
        self.tracklet_associations: Dict[str, TrackletGTAssociation] = {}
        self.obs_id_to_gt: Dict[str, int] = {}
        self.gt_to_obs_ids: Dict[int, List[str]] = {}

        self._load_gt()

    def _load_gt(self) -> None:
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"Ground-truth directory not found: {self.gt_dir}")

        for gt_file in sorted(self.gt_dir.glob("*gt*.txt")):
            cam_short = gt_file.name.split("_")[0].lower()  # e.g. c001
            long_cam_id = f"CAM_S01_C{cam_short[1:].upper()}"
            content = gt_file.read_bytes()
            self.file_hashes[cam_short] = hashlib.sha256(content).hexdigest()

            cam_frames: Dict[int, List[GroundTruthDetection]] = {}
            cam_ids: Set[int] = set()

            lines = content.decode("utf-8").strip().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                # MOTChallenge format: frame, ID, left, top, width, height, 1, -1, -1, -1
                raw_frame = int(parts[0])
                # Note: MOTChallenge frame is 1-based; Member 1 perception frame is 0-based
                f_0 = raw_frame - 1
                veh_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                bbox = [x, y, x + w, y + h]

                det = GroundTruthDetection(
                    camera_id=long_cam_id,
                    frame_id=f_0,
                    gt_vehicle_id=veh_id,
                    bbox=bbox,
                )
                cam_frames.setdefault(f_0, []).append(det)
                cam_ids.add(veh_id)
                self.gt_vehicle_ids.add(veh_id)

            self.gt_by_frame[long_cam_id] = cam_frames
            self.gt_by_frame[cam_short] = cam_frames
            self.gt_vehicles_by_camera[long_cam_id] = cam_ids

    def link_member1_tracklets(
        self,
        observations: List[Observation],
        handoff_output_dir: Union[str, Path] = "UrbanTrack_Member1_Handoff/output",
    ) -> Dict[str, TrackletGTAssociation]:
        """
        Independently match Member-1 tracklets to official ground-truth identities using
        spatiotemporal frame overlap and consensus IoU matching.

        Does NOT touch or mutate inference fields on Observation instances.
        """
        p_out = Path(handoff_output_dir)
        self.tracklet_associations.clear()
        self.obs_id_to_gt.clear()
        self.gt_to_obs_ids.clear()

        # Load frame-level detections per camera
        cam_obs_data: Dict[str, Dict[str, Any]] = {}
        for cam_id in set(o.camera_id for o in observations):
            obs_file = p_out / cam_id / "observations.json"
            if obs_file.is_file():
                with open(obs_file, "r", encoding="utf-8") as f:
                    cam_obs_data[cam_id] = json.load(f)

        # Group observations by camera and track_id
        for obs in observations:
            cam_id = obs.camera_id
            track_id = int(obs.track_id) if obs.track_id is not None else 0
            obs_json = cam_obs_data.get(cam_id)
            gt_cam_frames = self.gt_by_frame.get(cam_id, {})

            if not obs_json or not gt_cam_frames:
                assoc = TrackletGTAssociation(
                    observation_id=obs.observation_id,
                    camera_id=cam_id,
                    local_track_id=track_id,
                    gt_vehicle_id=None,
                    match_status="unmatched",
                    frames_evaluated=0,
                    overlapping_frames=0,
                    consensus_ratio=0.0,
                    mean_iou=0.0,
                )
                self.tracklet_associations[obs.observation_id] = assoc
                continue

            # Find all frames where this track was detected by Member 1
            matched_gt_ids: List[int] = []
            matched_ious: List[float] = []
            total_frames = 0

            for frame_dict in obs_json.get("frames", []):
                f_num = frame_dict.get("frame_number")
                veh_list = frame_dict.get("vehicles", [])
                target_veh = next((v for v in veh_list if v.get("track_id") == track_id), None)
                if target_veh:
                    total_frames += 1
                    gt_dets = gt_cam_frames.get(f_num, [])
                    t_bbox = target_veh.get("bbox")
                    if t_bbox and gt_dets:
                        best_iou = 0.0
                        best_gt_id = None
                        for gd in gt_dets:
                            score = compute_iou(t_bbox, gd.bbox)
                            if score > best_iou:
                                best_iou = score
                                best_gt_id = gd.gt_vehicle_id

                        if best_iou >= self.iou_threshold and best_gt_id is not None:
                            matched_gt_ids.append(best_gt_id)
                            matched_ious.append(best_iou)

            if matched_gt_ids:
                counter = Counter(matched_gt_ids)
                top_id, count = counter.most_common(1)[0]
                ratio = count / len(matched_gt_ids)
                mean_iou = sum(matched_ious) / len(matched_ious)

                # Check consensus requirement
                if ratio >= self.consensus_threshold:
                    status = "matched"
                    assigned_gt = top_id
                else:
                    status = "ambiguous"
                    assigned_gt = None

                competing = dict(counter)
            else:
                assigned_gt = None
                status = "unmatched"
                ratio = 0.0
                mean_iou = 0.0
                competing = {}

            assoc = TrackletGTAssociation(
                observation_id=obs.observation_id,
                camera_id=cam_id,
                local_track_id=track_id,
                gt_vehicle_id=assigned_gt,
                match_status=status,
                frames_evaluated=total_frames,
                overlapping_frames=len(matched_gt_ids),
                consensus_ratio=round(ratio, 4),
                mean_iou=round(mean_iou, 4),
                competing_ids=competing,
            )
            self.tracklet_associations[obs.observation_id] = assoc

            if assigned_gt is not None:
                self.obs_id_to_gt[obs.observation_id] = assigned_gt
                self.gt_to_obs_ids.setdefault(assigned_gt, []).append(obs.observation_id)

        return self.tracklet_associations

    def get_ground_truth_pair_label(
        self,
        obs_a_id: str,
        obs_b_id: str,
    ) -> Optional[bool]:
        """
        Return independent pairwise ground-truth label:
        - True:  Both observations mapped to the same official GT vehicle.
        - False: Both observations mapped to confirmed different official GT vehicles.
        - None:  One or both observations are unmatched or ambiguous (excluded from GT evaluation).
        """
        gt_a = self.obs_id_to_gt.get(obs_a_id)
        gt_b = self.obs_id_to_gt.get(obs_b_id)
        if gt_a is None or gt_b is None:
            return None
        return gt_a == gt_b

    def get_cross_camera_gt_pairs(
        self,
        observations: Optional[List[Observation]] = None,
    ) -> Set[Tuple[str, str]]:
        """
        Return all true positive cross-camera pairs (obs_a_id, obs_b_id) where
        both observations genuinely belong to the same physical vehicle across different cameras.
        """
        obs_cams: Dict[str, str] = {}
        if observations:
            for o in observations:
                obs_cams[o.observation_id] = o.camera_id
        else:
            for oid in self.obs_id_to_gt.keys():
                obs_cams[oid] = oid.split("_trk_")[0]

        true_cross_pairs: Set[Tuple[str, str]] = set()
        for gt_id, oids in self.gt_to_obs_ids.items():
            for i in range(len(oids)):
                for j in range(i + 1, len(oids)):
                    oa, ob = oids[i], oids[j]
                    if obs_cams.get(oa) != obs_cams.get(ob):
                        pair_key = (min(oa, ob), max(oa, ob))
                        true_cross_pairs.add(pair_key)

        return true_cross_pairs

    def to_dict(self) -> Dict[str, Any]:
        """Summary inventory of official ground-truth mapping."""
        multi_cam_gt = {
            gid: oids for gid, oids in self.gt_to_obs_ids.items()
            if len(set(o.split("_trk_")[0] for o in oids)) > 1
        }
        return {
            "total_official_gt_vehicles": len(self.gt_vehicle_ids),
            "mapped_observations_count": len(self.obs_id_to_gt),
            "unmatched_observations_count": sum(1 for a in self.tracklet_associations.values() if a.match_status == "unmatched"),
            "ambiguous_observations_count": sum(1 for a in self.tracklet_associations.values() if a.match_status == "ambiguous"),
            "mapped_gt_vehicles_count": len(self.gt_to_obs_ids),
            "multi_camera_gt_vehicles_count": len(multi_cam_gt),
            "file_hashes": self.file_hashes,
        }
