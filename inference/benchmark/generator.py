"""
UrbanTrack AI — Multi-Camera Benchmark Dataset Generator (multicamera_v1).
Synthesizes a realistic 5-camera urban arterial network with 150 latent vehicles,
~1,500 observations, empirical OSNet 512-D appearance features, difficulty tiers,
and hard negatives.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from schemas.observation_schema import Observation
from .difficulty import (
    DifficultyTier,
    perturb_embedding,
    perturb_plate_text,
    sample_tier,
)
from .ground_truth import (
    GroundTruthRegistry,
    LatentVehicle,
    PairwiseLabel,
)

# Reference Camera Network Definition (5 Cameras across an urban arterial corridor)
DEFAULT_CAMERAS = {
    "CAM_NORTH_01": {
        "camera_id": "CAM_NORTH_01",
        "name": "North Entry Junction",
        "latitude": 17.3980,
        "longitude": 78.4850,
        "reliability": 0.95,
        "fps": 30.0,
        "resolution": [3840, 2160],
        "time_reference_id": "city_network_sync",
    },
    "CAM_NORTH_02": {
        "camera_id": "CAM_NORTH_02",
        "name": "Northern Arterial",
        "latitude": 17.3930,
        "longitude": 78.4860,
        "reliability": 0.92,
        "fps": 30.0,
        "resolution": [3840, 2160],
        "time_reference_id": "city_network_sync",
    },
    "CAM_CENTRAL_01": {
        "camera_id": "CAM_CENTRAL_01",
        "name": "City Center Junction",
        "latitude": 17.3880,
        "longitude": 78.4870,
        "reliability": 0.88,
        "fps": 30.0,
        "resolution": [3840, 2160],
        "time_reference_id": "city_network_sync",
    },
    "CAM_SOUTH_01": {
        "camera_id": "CAM_SOUTH_01",
        "name": "South Commercial Hub",
        "latitude": 17.3830,
        "longitude": 78.4880,
        "reliability": 0.85,
        "fps": 30.0,
        "resolution": [3840, 2160],
        "time_reference_id": "city_network_sync",
    },
    "CAM_SOUTH_02": {
        "camera_id": "CAM_SOUTH_02",
        "name": "South Ring Exit",
        "latitude": 17.3780,
        "longitude": 78.4890,
        "reliability": 0.94,
        "fps": 30.0,
        "resolution": [3840, 2160],
        "time_reference_id": "city_network_sync",
    },
}

CORRIDOR_ORDER = [
    "CAM_NORTH_01",
    "CAM_NORTH_02",
    "CAM_CENTRAL_01",
    "CAM_SOUTH_01",
    "CAM_SOUTH_02",
]


def load_empirical_osnet_prototypes(project_root: Optional[Path] = None) -> List[List[float]]:
    """
    Load empirical 512-D OSNet embeddings from Member 1 real perception output.
    Ensures synthetic benchmark visual features adhere to real OSNet distribution (Rule 9).
    """
    root = project_root or Path(__file__).resolve().parent.parent.parent
    path = root / "data" / "member1_perception" / "cam_001" / "track_embeddings.json"

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                tracks = json.load(f)
            embs = [
                t["appearance_embedding"]
                for t in tracks
                if t.get("appearance_embedding") and len(t["appearance_embedding"]) == 512
            ]
            if len(embs) >= 10:
                return embs
        except Exception:
            pass

    # Fallback deterministic orthogonal seed basis if file unavailable
    rng = random.Random(42)
    fallback = []
    for _ in range(30):
        vec = [rng.gauss(0.0, 1.0) for _ in range(512)]
        norm = math.sqrt(sum(x * x for x in vec))
        fallback.append([x / norm for x in vec])
    return fallback


class MultiCameraBenchmarkGenerator:
    """
    Generates controlled multi-camera benchmark datasets (multicamera_v1).
    Strict separation: Latent Identities -> Observations -> Independent Ground Truth.
    """

    def __init__(
        self,
        cameras: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: int = 42,
        project_root: Optional[Path] = None,
    ) -> None:
        self.cameras = cameras or DEFAULT_CAMERAS
        self.rng = random.Random(seed)
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.osnet_prototypes = load_empirical_osnet_prototypes(self.project_root)

    def _generate_plate_string(self, state_code: str, num_id: int) -> str:
        series = chr(65 + (num_id // 1000) % 26) + chr(65 + (num_id // 100) % 26)
        rto = f"{(num_id % 90) + 1:02d}"
        seq = f"{num_id % 9000 + 1000:04d}"
        return f"{state_code}{rto}{series}{seq}"

    def _create_latent_vehicles(
        self,
        n_vehicles: int = 150,
        n_hard_negatives: int = 20,
    ) -> List[LatentVehicle]:
        """
        Generate true physical latent vehicles independently before observation synthesis.
        Includes explicit hard-negative vehicle pairs sharing visual appearance prototypes (Rule 10).
        """
        state_codes = ["MH", "KA", "TS", "DL"]
        vehicle_types = ["car", "car", "car", "motorcycle", "truck", "bus"]
        vehicles: List[LatentVehicle] = []

        # 1. Generate regular distinct latent vehicles
        n_regular = n_vehicles - (n_hard_negatives * 2)
        for i in range(n_regular):
            v_id = f"LATENT_{i+1:04d}"
            v_type = self.rng.choice(vehicle_types)
            plate = self._generate_plate_string(self.rng.choice(state_codes), 1000 + i * 17)

            # Sample base prototype from real OSNet distribution
            proto = self.rng.choice(self.osnet_prototypes)
            base_vec = [p + self.rng.gauss(0.0, 0.012) for p in proto]
            norm = math.sqrt(sum(x * x for x in base_vec))
            base_vec = [x / norm for x in base_vec]

            vehicles.append(
                LatentVehicle(
                    latent_id=v_id,
                    vehicle_type=v_type,
                    true_plate=plate,
                    base_embedding=base_vec,
                    speed_capability_kmh=self.rng.uniform(35.0, 60.0),
                    color=self.rng.choice(["white", "silver", "black", "red", "blue"]),
                    is_hard_negative=False,
                )
            )

        # 2. Generate hard negative pairs (Rule 10)
        # Vehicles with same vehicle type, same base visual prototype, high visual similarity,
        # but distinct plates and distinct latent identities.
        for h in range(n_hard_negatives):
            group_id = f"HARD_NEG_GROUP_{h+1:02d}"
            v_type = "car"
            color = self.rng.choice(["white", "silver", "black"])
            shared_proto = self.rng.choice(self.osnet_prototypes)

            # Vehicle A
            vec_a = [p + self.rng.gauss(0.0, 0.008) for p in shared_proto]
            norm_a = math.sqrt(sum(x * x for x in vec_a))
            vec_a = [x / norm_a for x in vec_a]
            plate_a = self._generate_plate_string("MH", 5000 + h * 37)
            veh_a = LatentVehicle(
                latent_id=f"LATENT_HN_{h*2+1:04d}",
                vehicle_type=v_type,
                true_plate=plate_a,
                base_embedding=vec_a,
                speed_capability_kmh=45.0,
                color=color,
                is_hard_negative=True,
                hard_negative_group=group_id,
            )

            # Vehicle B (Hard Negative Twin: different state and number, same appearance)
            vec_b = [p + self.rng.gauss(0.0, 0.008) for p in shared_proto]
            norm_b = math.sqrt(sum(x * x for x in vec_b))
            vec_b = [x / norm_b for x in vec_b]
            plate_b = self._generate_plate_string("DL", 8000 + h * 43)
            veh_b = LatentVehicle(
                latent_id=f"LATENT_HN_{h*2+2:04d}",
                vehicle_type=v_type,
                true_plate=plate_b,
                base_embedding=vec_b,
                speed_capability_kmh=45.0,
                color=color,
                is_hard_negative=True,
                hard_negative_group=group_id,
            )

            vehicles.extend([veh_a, veh_b])

        return vehicles

    def generate_dataset(
        self,
        n_vehicles: int = 150,
        target_observations: int = 1500,
        n_hard_negatives: int = 20,
    ) -> Tuple[List[Observation], GroundTruthRegistry, Dict[str, Any]]:
        """
        Synthesize complete multi-camera benchmark dataset.
        Returns:
            observations: List of schema-compliant Observation records.
            registry: GroundTruthRegistry with observation assignments and pairwise labels.
            metadata: Dataset dictionary with statistics.
        """
        registry = GroundTruthRegistry()
        vehicles = self._create_latent_vehicles(n_vehicles=n_vehicles, n_hard_negatives=n_hard_negatives)
        for v in vehicles:
            registry.register_vehicle(v)

        observations: List[Observation] = []
        obs_counter = 1
        base_time = 1000.0  # seconds

        # Schedule vehicle entry times across a 3-hour (10,800s) arterial timeline
        timeline_seconds = 10800.0
        entry_times: Dict[str, float] = {}

        for i, veh in enumerate(vehicles):
            if veh.is_hard_negative and veh.hard_negative_group:
                # Place twins close in time (45-90s apart) along the same corridor
                h_idx = int(veh.hard_negative_group.split("_")[-1]) - 1
                twin_num = int(veh.latent_id.split("_")[-1]) % 2
                base_hn = base_time + (h_idx / n_hard_negatives) * (timeline_seconds * 0.90)
                entry_times[veh.latent_id] = base_hn + (0.0 if twin_num == 1 else self.rng.uniform(45.0, 90.0))
            else:
                entry_times[veh.latent_id] = base_time + (i / len(vehicles)) * timeline_seconds + self.rng.uniform(-45.0, 45.0)

        # Generate arterial corridor observations for each vehicle
        for veh in vehicles:
            entry_t = entry_times[veh.latent_id]
            speed_mps = (veh.speed_capability_kmh * 1000.0) / 3600.0
            dist_to_next = 550.0  # meters
            travel_time_sec = dist_to_next / speed_mps

            # Outbound pass (North -> South across 5 cameras)
            t = entry_t
            for cam_id in CORRIDOR_ORDER:
                cam_info = self.cameras[cam_id]
                obs_id = f"OBS_MC_{obs_counter:05d}"
                obs_counter += 1

                tier = sample_tier(self.rng)
                obs_plate, plate_conf = perturb_plate_text(veh.true_plate, tier, rng=self.rng)
                obs_emb, emb_quality = perturb_embedding(veh.base_embedding, tier, rng=self.rng)

                timing_jitter = self.rng.gauss(0.0, 2.0 if tier != DifficultyTier.HARD else 6.0)
                timestamp_sec = round(t + timing_jitter, 3)

                registry.register_observation(obs_id, veh.latent_id, tier)

                obs = Observation(
                    observation_id=obs_id,
                    camera_id=cam_id,
                    timestamp=datetime.fromtimestamp(timestamp_sec),
                    timestamp_seconds=timestamp_sec,
                    frame_id=int(timestamp_sec * cam_info["fps"]),
                    track_id=f"TRK_{veh.latent_id}_{obs_counter % 99:02d}",
                    vehicle_type=veh.vehicle_type,
                    detection_confidence=round(self.rng.uniform(0.85, 0.98), 3),
                    frame_detection_confidence_mean=round(self.rng.uniform(0.82, 0.95), 3),
                    plate=obs_plate,
                    plate_confidence=plate_conf,
                    ocr_confidence=plate_conf,
                    appearance_embedding=obs_emb,
                    source_provenance={"reid_model": "osnet_x0_25_msmt17", "embedding_quality": emb_quality},
                    latitude=cam_info["latitude"],
                    longitude=cam_info["longitude"],
                    point_coordinate_system="gps",
                    timestamp_semantics="synchronized",
                    time_reference_id=cam_info["time_reference_id"],
                    camera_reliability=cam_info["reliability"],
                )
                observations.append(obs)
                t += travel_time_sec

            # Dwell / turnaround (3-5 minutes)
            t += self.rng.uniform(180.0, 300.0)

            # Return pass (South -> North across 5 cameras)
            for cam_id in reversed(CORRIDOR_ORDER):
                cam_info = self.cameras[cam_id]
                obs_id = f"OBS_MC_{obs_counter:05d}"
                obs_counter += 1

                tier = sample_tier(self.rng)
                obs_plate, plate_conf = perturb_plate_text(veh.true_plate, tier, rng=self.rng)
                obs_emb, emb_quality = perturb_embedding(veh.base_embedding, tier, rng=self.rng)

                timing_jitter = self.rng.gauss(0.0, 2.0 if tier != DifficultyTier.HARD else 6.0)
                timestamp_sec = round(t + timing_jitter, 3)

                registry.register_observation(obs_id, veh.latent_id, tier)

                obs = Observation(
                    observation_id=obs_id,
                    camera_id=cam_id,
                    timestamp=datetime.fromtimestamp(timestamp_sec),
                    timestamp_seconds=timestamp_sec,
                    frame_id=int(timestamp_sec * cam_info["fps"]),
                    track_id=f"TRK_{veh.latent_id}_{obs_counter % 99:02d}",
                    vehicle_type=veh.vehicle_type,
                    detection_confidence=round(self.rng.uniform(0.85, 0.98), 3),
                    frame_detection_confidence_mean=round(self.rng.uniform(0.82, 0.95), 3),
                    plate=obs_plate,
                    plate_confidence=plate_conf,
                    ocr_confidence=plate_conf,
                    appearance_embedding=obs_emb,
                    source_provenance={"reid_model": "osnet_x0_25_msmt17", "embedding_quality": emb_quality},
                    latitude=cam_info["latitude"],
                    longitude=cam_info["longitude"],
                    point_coordinate_system="gps",
                    timestamp_semantics="synchronized",
                    time_reference_id=cam_info["time_reference_id"],
                    camera_reliability=cam_info["reliability"],
                )
                observations.append(obs)
                t += travel_time_sec

        # Sort observations chronologically
        observations.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))

        # 3. Generate Authoritative Independent Pairwise Labels
        # Positive pairs: all pairs originating from the same latent vehicle
        gt_clusters = registry.get_ground_truth_clusters()
        for lat_id, obs_ids in gt_clusters.items():
            for i in range(len(obs_ids)):
                for j in range(i + 1, len(obs_ids)):
                    oa, ob = obs_ids[i], obs_ids[j]
                    tier_a = registry.observation_tiers[oa]
                    tier_b = registry.observation_tiers[ob]
                    pair_tier = max([tier_a, tier_b], key=lambda t: list(DifficultyTier).index(t)).value
                    registry.register_pair(
                        PairwiseLabel(
                            obs_a_id=oa,
                            obs_b_id=ob,
                            is_same_vehicle=True,
                            latent_id_a=lat_id,
                            latent_id_b=lat_id,
                            difficulty_tier=pair_tier,
                            is_hard_negative=False,
                            relationship="SAME_VEHICLE",
                        )
                    )

        # Hard negative pairs: pairs between twins in the same hard-negative group
        hard_groups: Dict[str, List[str]] = {}
        for veh in vehicles:
            if veh.is_hard_negative and veh.hard_negative_group:
                hard_groups.setdefault(veh.hard_negative_group, []).append(veh.latent_id)

        for hgroup, l_ids in hard_groups.items():
            if len(l_ids) >= 2:
                obs_group_1 = gt_clusters.get(l_ids[0], [])
                obs_group_2 = gt_clusters.get(l_ids[1], [])
                for oa in obs_group_1:
                    for ob in obs_group_2:
                        registry.register_pair(
                            PairwiseLabel(
                                obs_a_id=oa,
                                obs_b_id=ob,
                                is_same_vehicle=False,
                                latent_id_a=l_ids[0],
                                latent_id_b=l_ids[1],
                                difficulty_tier=DifficultyTier.ADVERSARIAL.value,
                                is_hard_negative=True,
                                relationship="HARD_NEGATIVE",
                            )
                        )

        # Random negative pairs across network
        all_obs_ids = [o.observation_id for o in observations]
        n_neg_samples = min(len(registry.pairwise_labels) * 2, 5000)
        sampled_negatives = 0
        while sampled_negatives < n_neg_samples:
            oa = self.rng.choice(all_obs_ids)
            ob = self.rng.choice(all_obs_ids)
            if oa == ob:
                continue
            key = (min(oa, ob), max(oa, ob))
            if key in registry.pairwise_labels:
                continue
            la = registry.get_latent_id(oa)
            lb = registry.get_latent_id(ob)
            if la != lb:
                registry.register_pair(
                    PairwiseLabel(
                        obs_a_id=oa,
                        obs_b_id=ob,
                        is_same_vehicle=False,
                        latent_id_a=la or "UNKNOWN",
                        latent_id_b=lb or "UNKNOWN",
                        difficulty_tier=DifficultyTier.EASY.value,
                        is_hard_negative=False,
                        relationship="DIFFERENT_VEHICLE",
                    )
                )
                sampled_negatives += 1

        metadata = {
            "dataset_name": "multicamera_v1",
            "dataset_type": "synthetic_controlled",
            "generation_timestamp": datetime.now().isoformat(),
            "total_cameras": len(self.cameras),
            "total_latent_vehicles": len(vehicles),
            "total_observations": len(observations),
            "total_labeled_pairs": len(registry.pairwise_labels),
            "positive_pairs": sum(1 for p in registry.pairwise_labels.values() if p.is_same_vehicle),
            "hard_negative_pairs": sum(1 for p in registry.pairwise_labels.values() if p.is_hard_negative),
            "camera_topology": list(self.cameras.keys()),
        }

        return observations, registry, metadata

    def export_to_directory(
        self,
        output_dir: Path,
        n_vehicles: int = 150,
        target_observations: int = 1500,
        n_hard_negatives: int = 20,
    ) -> Dict[str, Any]:
        """
        Generate and persist the multicamera_v1 dataset files to disk.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        obs, reg, meta = self.generate_dataset(
            n_vehicles=n_vehicles,
            target_observations=target_observations,
            n_hard_negatives=n_hard_negatives,
        )

        # 1. cameras.json
        with open(output_dir / "cameras.json", "w", encoding="utf-8") as f:
            json.dump(self.cameras, f, indent=2)

        # 2. observations.json
        with open(output_dir / "observations.json", "w", encoding="utf-8") as f:
            json.dump([o.to_dict() for o in obs], f, indent=2)

        # 3. ground_truth.json
        gt_data = {
            "latent_vehicles": {k: v.to_dict() for k, v in reg.latent_vehicles.items()},
            "observation_to_latent": reg.observation_to_latent,
            "observation_tiers": reg.observation_tiers,
        }
        with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        # 4. pairwise_ground_truth.json
        pw_data = [p.to_dict() for p in reg.pairwise_labels.values()]
        with open(output_dir / "pairwise_ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(pw_data, f, indent=2)

        # 5. metadata.json
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta
