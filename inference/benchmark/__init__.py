"""
UrbanTrack AI — Independent Multi-Camera Benchmark Package.
"""

from .difficulty import (
    DifficultyTier,
    OCR_CONFUSION_MAP,
    perturb_embedding,
    perturb_plate_text,
    sample_tier,
)
from .ground_truth import (
    GroundTruthRegistry,
    LatentVehicle,
    PairwiseLabel,
)
from .generator import (
    CORRIDOR_ORDER,
    DEFAULT_CAMERAS,
    MultiCameraBenchmarkGenerator,
    load_empirical_osnet_prototypes,
)
from .evaluator import (
    BenchmarkEvaluationResult,
    MultiCameraBenchmarkEvaluator,
)
from .runner import (
    run_multicamera_benchmark,
)

__all__ = [
    "DifficultyTier",
    "OCR_CONFUSION_MAP",
    "perturb_embedding",
    "perturb_plate_text",
    "sample_tier",
    "GroundTruthRegistry",
    "LatentVehicle",
    "PairwiseLabel",
    "DEFAULT_CAMERAS",
    "CORRIDOR_ORDER",
    "MultiCameraBenchmarkGenerator",
    "load_empirical_osnet_prototypes",
    "BenchmarkEvaluationResult",
    "MultiCameraBenchmarkEvaluator",
    "run_multicamera_benchmark",
]
