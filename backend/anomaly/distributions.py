"""Numerically stable distribution comparisons for Phase 5."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple, TypeVar

from backend.anomaly.models import DistributionKey, DistributionShift

Key = TypeVar("Key")


def normalize_distribution(values: Mapping[Key, float]) -> Dict[Key, float]:
    """Return a deterministic probability mass mapping without mutating input."""
    if not isinstance(values, Mapping):
        raise TypeError("distribution must be a mapping")
    cleaned: Dict[Key, float] = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"distribution value for {key!r} must be finite and non-negative")
        if value > 0.0:
            cleaned[key] = float(value)
    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    return {key: cleaned[key] / total for key in sorted(cleaned, key=repr)}


def jensen_shannon_divergence(
    baseline: Mapping[Key, float],
    current: Mapping[Key, float],
) -> float:
    """Calculate symmetric, bounded Jensen-Shannon divergence using log base 2."""
    baseline_norm = normalize_distribution(baseline)
    current_norm = normalize_distribution(current)
    if not baseline_norm and not current_norm:
        return 0.0
    keys = sorted(set(baseline_norm) | set(current_norm), key=repr)
    midpoint = {
        key: (baseline_norm.get(key, 0.0) + current_norm.get(key, 0.0)) / 2.0
        for key in keys
    }

    def kl_divergence(distribution: Mapping[Key, float]) -> float:
        total = 0.0
        for key, probability in distribution.items():
            if probability > 0.0:
                total += probability * math.log2(probability / midpoint[key])
        return total

    return max(0.0, min(1.0, 0.5 * (kl_divergence(baseline_norm) + kl_divergence(current_norm))))


def compare_distributions(
    baseline: Mapping[DistributionKey, float],
    current: Mapping[DistributionKey, float],
    threshold: float,
) -> DistributionShift:
    """Return normalized distributions, divergence, and threshold status."""
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be a finite non-negative number")
    baseline_norm = normalize_distribution(baseline)
    current_norm = normalize_distribution(current)
    divergence = jensen_shannon_divergence(baseline_norm, current_norm)
    return DistributionShift(
        divergence=divergence,
        baseline_distribution=baseline_norm,
        current_distribution=current_norm,
        threshold_exceeded=divergence >= float(threshold),
    )
