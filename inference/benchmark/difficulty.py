"""
UrbanTrack AI — Multi-Camera Benchmark Difficulty Tiers & Noise Models.
Defines difficulty categories and perturbation mechanics for controlled benchmark synthesis.
"""

from enum import Enum
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class DifficultyTier(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    ADVERSARIAL = "ADVERSARIAL"


# Character confusion matrix for realistic OCR errors on Indian registration plates
OCR_CONFUSION_MAP = {
    "0": ["O", "D", "Q"],
    "O": ["0", "D", "Q"],
    "D": ["0", "O"],
    "1": ["I", "L", "T"],
    "I": ["1", "L"],
    "8": ["B", "S", "3"],
    "B": ["8", "3"],
    "5": ["S"],
    "S": ["5", "8"],
    "2": ["Z"],
    "Z": ["2"],
    "6": ["G", "C"],
    "G": ["6", "C"],
    "A": ["4"],
    "4": ["A"],
}


def perturb_plate_text(
    plate: str,
    tier: DifficultyTier,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Apply realistic OCR noise based on difficulty tier.

    Returns:
        Tuple[Optional[str], Optional[float]]: (observed_plate, plate_confidence)
    """
    r = rng or random

    if tier == DifficultyTier.EASY:
        # 100% plate availability, high confidence [0.90, 0.99], 0 character mutations
        return plate, round(r.uniform(0.90, 0.99), 3)

    elif tier == DifficultyTier.MEDIUM:
        # 95% availability, 1 character substitution with 25% probability
        if r.random() < 0.05:
            return None, None
        chars = list(plate)
        if r.random() < 0.25 and len(chars) > 4:
            # Mutate one alphanumeric character
            mut_idx = r.randint(2, len(chars) - 1)
            char = chars[mut_idx]
            if char in OCR_CONFUSION_MAP:
                chars[mut_idx] = r.choice(OCR_CONFUSION_MAP[char])
        return "".join(chars), round(r.uniform(0.72, 0.88), 3)

    elif tier == DifficultyTier.HARD:
        # 30% plate completely missing (occlusion, high speed, extreme angle)
        if r.random() < 0.30:
            return None, None
        chars = list(plate)
        # 1-2 character errors
        mut_count = r.choice([1, 2])
        for _ in range(mut_count):
            mut_idx = r.randint(0, len(chars) - 1)
            char = chars[mut_idx]
            if char in OCR_CONFUSION_MAP:
                chars[mut_idx] = r.choice(OCR_CONFUSION_MAP[char])
        return "".join(chars), round(r.uniform(0.50, 0.70), 3)

    elif tier == DifficultyTier.ADVERSARIAL:
        # 40% missing or misleading plate
        chance = r.random()
        if chance < 0.35:
            return None, None
        elif chance < 0.60:
            # Heavy OCR degradation (multiple character errors)
            chars = list(plate)
            for i in range(len(chars)):
                if r.random() < 0.30 and chars[i] in OCR_CONFUSION_MAP:
                    chars[i] = r.choice(OCR_CONFUSION_MAP[chars[i]])
            return "".join(chars), round(r.uniform(0.35, 0.60), 3)
        else:
            return plate, round(r.uniform(0.60, 0.80), 3)

    return plate, 0.90


def perturb_embedding(
    base_embedding: List[float],
    tier: DifficultyTier,
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[List[float]], float]:
    """
    Apply Gaussian perturbation to OSNet 512-D unit embedding based on difficulty tier.
    Preserves L2 normalization.

    Returns:
        Tuple[Optional[List[float]], float]: (perturbed_unit_vector, embedding_quality)
    """
    r = rng or random

    if tier == DifficultyTier.EASY:
        # Very low noise: expected cosine similarity ~0.94 - 0.98 in 512-D
        noise_std = 0.011
        quality = r.uniform(0.88, 0.98)
    elif tier == DifficultyTier.MEDIUM:
        # Moderate noise: expected cosine similarity ~0.82 - 0.88 in 512-D
        noise_std = 0.0185
        quality = r.uniform(0.72, 0.88)
    elif tier == DifficultyTier.HARD:
        # 15% probability of missing embedding (severe occlusion/crop failure)
        if r.random() < 0.15:
            return None, 0.0
        # High noise: expected cosine similarity ~0.65 - 0.74 in 512-D
        noise_std = 0.0275
        quality = r.uniform(0.50, 0.70)
    elif tier == DifficultyTier.ADVERSARIAL:
        # 25% missing embedding
        if r.random() < 0.25:
            return None, 0.0
        # Extreme noise / lighting shift: expected cosine ~0.55 - 0.65 in 512-D
        noise_std = 0.0376
        quality = r.uniform(0.35, 0.55)
    else:
        noise_std = 0.015
        quality = 0.90

    perturbed = [x + r.gauss(0.0, noise_std) for x in base_embedding]
    norm = math.sqrt(sum(x * x for x in perturbed))
    if norm < 1e-12:
        return [1.0 / math.sqrt(len(base_embedding))] * len(base_embedding), quality
    normalized = [round(x / norm, 6) for x in perturbed]
    return normalized, round(quality, 3)


def get_tier_distribution(tier_counts: Optional[Dict[DifficultyTier, float]] = None) -> List[Tuple[DifficultyTier, float]]:
    """
    Default difficulty tier distribution:
    - 30% EASY
    - 40% MEDIUM
    - 20% HARD
    - 10% ADVERSARIAL
    """
    if tier_counts:
        return list(tier_counts.items())
    return [
        (DifficultyTier.EASY, 0.30),
        (DifficultyTier.MEDIUM, 0.40),
        (DifficultyTier.HARD, 0.20),
        (DifficultyTier.ADVERSARIAL, 0.10),
    ]


def sample_tier(rng: Optional[random.Random] = None) -> DifficultyTier:
    """Sample a difficulty tier according to target distribution."""
    r = rng or random
    p = r.random()
    if p < 0.30:
        return DifficultyTier.EASY
    elif p < 0.70:
        return DifficultyTier.MEDIUM
    elif p < 0.90:
        return DifficultyTier.HARD
    else:
        return DifficultyTier.ADVERSARIAL
