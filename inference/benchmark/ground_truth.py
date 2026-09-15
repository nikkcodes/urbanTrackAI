"""
UrbanTrack AI — Independent Multi-Camera Benchmark Ground Truth Infrastructure.
Ensures ground truth is generated from latent vehicle identities BEFORE any observation or matching occurs.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .difficulty import DifficultyTier


@dataclass
class LatentVehicle:
    """
    True physical latent vehicle entity.
    Generated independently before any camera observations are synthesized.
    """
    latent_id: str
    vehicle_type: str
    true_plate: str
    base_embedding: List[float]
    speed_capability_kmh: float = 60.0
    color: str = "white"
    is_hard_negative: bool = False
    hard_negative_group: Optional[str] = None
    assigned_corridor: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "vehicle_type": self.vehicle_type,
            "true_plate": self.true_plate,
            "embedding_dim": len(self.base_embedding),
            "speed_capability_kmh": self.speed_capability_kmh,
            "color": self.color,
            "is_hard_negative": self.is_hard_negative,
            "hard_negative_group": self.hard_negative_group,
            "assigned_corridor": self.assigned_corridor,
        }


@dataclass
class PairwiseLabel:
    """
    Authoritative independent pairwise relationship between two observations.
    """
    obs_a_id: str
    obs_b_id: str
    is_same_vehicle: bool
    latent_id_a: str
    latent_id_b: str
    difficulty_tier: str
    is_hard_negative: bool = False
    relationship: str = "DIFFERENT_VEHICLE"  # SAME_VEHICLE, HARD_NEGATIVE, DIFFERENT_VEHICLE

    def pair_key(self) -> Tuple[str, str]:
        return (min(self.obs_a_id, self.obs_b_id), max(self.obs_a_id, self.obs_b_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_a_id": self.obs_a_id,
            "obs_b_id": self.obs_b_id,
            "is_same_vehicle": self.is_same_vehicle,
            "latent_id_a": self.latent_id_a,
            "latent_id_b": self.latent_id_b,
            "difficulty_tier": self.difficulty_tier,
            "is_hard_negative": self.is_hard_negative,
            "relationship": self.relationship,
        }


class GroundTruthRegistry:
    """
    Independent registry storing observation-level and pairwise ground truth.
    Contains zero predictions, zero heuristic scores, and zero circular dependencies.
    """
    def __init__(self) -> None:
        self.latent_vehicles: Dict[str, LatentVehicle] = {}
        self.observation_to_latent: Dict[str, str] = {}
        self.observation_tiers: Dict[str, DifficultyTier] = {}
        self.pairwise_labels: Dict[Tuple[str, str], PairwiseLabel] = {}

    def register_vehicle(self, vehicle: LatentVehicle) -> None:
        self.latent_vehicles[vehicle.latent_id] = vehicle

    def register_observation(
        self,
        observation_id: str,
        latent_id: str,
        tier: DifficultyTier = DifficultyTier.EASY,
    ) -> None:
        if latent_id not in self.latent_vehicles:
            raise ValueError(f"Latent vehicle {latent_id} must be registered before observations.")
        self.observation_to_latent[observation_id] = latent_id
        self.observation_tiers[observation_id] = tier

    def register_pair(self, label: PairwiseLabel) -> None:
        key = label.pair_key()
        self.pairwise_labels[key] = label

    def is_same_vehicle(self, obs_a_id: str, obs_b_id: str) -> bool:
        la = self.observation_to_latent.get(obs_a_id)
        lb = self.observation_to_latent.get(obs_b_id)
        if la is None or lb is None:
            return False
        return la == lb

    def get_latent_id(self, observation_id: str) -> Optional[str]:
        return self.observation_to_latent.get(observation_id)

    def get_ground_truth_clusters(self) -> Dict[str, List[str]]:
        """Return mapping from latent_vehicle_id -> list of observation_ids."""
        clusters: Dict[str, List[str]] = {}
        for obs_id, lat_id in self.observation_to_latent.items():
            clusters.setdefault(lat_id, []).append(obs_id)
        return clusters

    def export_ground_truth_json(self, output_path: Path) -> None:
        """Export observation -> latent mapping and vehicle profiles."""
        data = {
            "metadata": {
                "dataset_name": "multicamera_v1",
                "ground_truth_type": "INDEPENDENT_LATENT_SYNTHESIS",
                "total_latent_vehicles": len(self.latent_vehicles),
                "total_observations": len(self.observation_to_latent),
            },
            "latent_vehicles": {vid: v.to_dict() for vid, v in self.latent_vehicles.items()},
            "observation_assignments": self.observation_to_latent,
            "observation_tiers": {k: v.value for k, v in self.observation_tiers.items()},
            "ground_truth_clusters": self.get_ground_truth_clusters(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_pairwise_ground_truth_json(self, output_path: Path) -> None:
        """Export pairwise ground-truth relationships."""
        pairs_list = [p.to_dict() for p in self.pairwise_labels.values()]
        data = {
            "metadata": {
                "dataset_name": "multicamera_v1",
                "total_labeled_pairs": len(self.pairwise_labels),
                "same_vehicle_pairs": sum(1 for p in self.pairwise_labels.values() if p.is_same_vehicle),
                "hard_negative_pairs": sum(1 for p in self.pairwise_labels.values() if p.is_hard_negative),
                "different_vehicle_pairs": sum(1 for p in self.pairwise_labels.values() if not p.is_same_vehicle),
            },
            "pairs": pairs_list,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_directory(cls, dir_path: Path) -> "GroundTruthRegistry":
        """Load registry from ground_truth.json and pairwise_ground_truth.json."""
        registry = cls()
        gt_path = dir_path / "ground_truth.json"
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        for vid, vdata in gt_data.get("latent_vehicles", {}).items():
            registry.register_vehicle(
                LatentVehicle(
                    latent_id=vid,
                    vehicle_type=vdata["vehicle_type"],
                    true_plate=vdata["true_plate"],
                    base_embedding=[0.0] * vdata.get("embedding_dim", 512),
                    speed_capability_kmh=vdata.get("speed_capability_kmh", 60.0),
                    color=vdata.get("color", "white"),
                    is_hard_negative=vdata.get("is_hard_negative", False),
                    hard_negative_group=vdata.get("hard_negative_group"),
                    assigned_corridor=vdata.get("assigned_corridor", []),
                )
            )

        obs_assignments = gt_data.get("observation_to_latent") or gt_data.get("observation_assignments", {})
        for obs_id, lat_id in obs_assignments.items():
            tier_str = gt_data.get("observation_tiers", {}).get(obs_id, "EASY")
            registry.register_observation(obs_id, lat_id, DifficultyTier(tier_str))

        pairwise_path = dir_path / "pairwise_ground_truth.json"
        if pairwise_path.exists():
            with open(pairwise_path, "r", encoding="utf-8") as f:
                pw_data = json.load(f)
            pairs_list = pw_data if isinstance(pw_data, list) else pw_data.get("pairs", [])
            for p in pairs_list:
                registry.register_pair(
                    PairwiseLabel(
                        obs_a_id=p["obs_a_id"],
                        obs_b_id=p["obs_b_id"],
                        is_same_vehicle=p["is_same_vehicle"],
                        latent_id_a=p["latent_id_a"],
                        latent_id_b=p["latent_id_b"],
                        difficulty_tier=p["difficulty_tier"],
                        is_hard_negative=p.get("is_hard_negative", False),
                        relationship=p.get("relationship", "SAME_VEHICLE" if p["is_same_vehicle"] else "DIFFERENT_VEHICLE"),
                    )
                )

        return registry
