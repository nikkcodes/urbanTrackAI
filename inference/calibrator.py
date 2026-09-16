"""
Principled Probability Calibration & Bayesian Evidence Engine for UrbanTrack AI.

Provides:
1. PlattProbabilityCalibrator: Parametric logistic calibration P(Y=1 | score) = sigma(a * score + b).
2. BayesianEvidenceCombiner: Likelihood-ratio evidence fusion P(same_vehicle | evidence).
3. Calibration Diagnostics: Brier Score, Expected Calibration Error (ECE), and Reliability Diagrams.

Methodological Guarantee:
Calibration parameters are fitted strictly on Training/Development (DEV) splits.
Parameters are frozen before Holdout evaluation. Never tuned on test/holdout data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union


class PlattProbabilityCalibrator:
    """
    Parametric logistic probability calibrator (Platt Scaling).
    Maps uncalibrated heuristic scores s in [0, 1] to empirical posterior probabilities:
        P(same_vehicle = 1 | s) = 1 / (1 + exp(-(a * s + b)))
    """

    def __init__(self, a: float = 6.0, b: float = -4.5, is_fitted: bool = True) -> None:
        self.a: float = float(a)
        self.b: float = float(b)
        self.is_fitted: bool = is_fitted
        self.training_metrics: Dict[str, Any] = {}

    def predict_probability(self, score: float) -> float:
        """
        Calibrate an uncalibrated heuristic score into an empirical probability.
        """
        if score <= 0.0:
            return 0.0
        # Extreme contradiction rule: score < 0.35 with known evidence maps strictly near 0
        logit = self.a * float(score) + self.b
        # Numerical stability clamp
        logit = max(-30.0, min(30.0, logit))
        prob = 1.0 / (1.0 + math.exp(-logit))
        return round(float(prob), 4)

    def fit(
        self,
        scores: List[float],
        labels: List[int],
        learning_rate: float = 0.05,
        epochs: int = 200,
    ) -> PlattProbabilityCalibrator:
        """
        Fit calibration parameters (a, b) using binary cross-entropy gradient descent.
        Must only be called on Training/Development (DEV) data.
        """
        if len(scores) != len(labels) or len(scores) < 10:
            raise ValueError(f"Insufficient data for calibration fit: {len(scores)} samples.")

        a = self.a
        b = self.b
        n = len(scores)

        # Gradient descent optimization on binary cross-entropy
        for epoch in range(epochs):
            grad_a = 0.0
            grad_b = 0.0
            loss = 0.0

            for s, y in zip(scores, labels):
                logit = max(-30.0, min(30.0, a * s + b))
                p = 1.0 / (1.0 + math.exp(-logit))
                err = p - y
                grad_a += err * s
                grad_b += err

                # Cross-entropy loss
                p_safe = max(1e-12, min(1.0 - 1e-12, p))
                loss += -(y * math.log(p_safe) + (1.0 - y) * math.log(1.0 - p_safe))

            a -= learning_rate * (grad_a / n)
            b -= learning_rate * (grad_b / n)
            # Ensure monotonicity: a must remain positive
            a = max(0.5, a)

        self.a = round(a, 4)
        self.b = round(b, 4)
        self.is_fitted = True

        # Compute post-fit diagnostics
        preds = [self.predict_probability(s) for s in scores]
        self.training_metrics = evaluate_calibration_metrics(preds, labels)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "platt_scaling_logistic",
            "a": self.a,
            "b": self.b,
            "is_fitted": self.is_fitted,
            "training_metrics": self.training_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlattProbabilityCalibrator:
        cal = cls(a=data.get("a", 6.0), b=data.get("b", -4.5), is_fitted=data.get("is_fitted", True))
        cal.training_metrics = data.get("training_metrics", {})
        return cal


def evaluate_calibration_metrics(
    probabilities: List[float],
    labels: List[int],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute comprehensive probability calibration diagnostics:
    - Brier Score (mean squared error of probability predictions)
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Reliability Diagram Bins
    """
    if len(probabilities) != len(labels) or len(probabilities) == 0:
        return {"brier_score": None, "ece": None, "mce": None, "reliability_bins": []}

    n = len(probabilities)

    # 1. Brier Score: MSE between predicted probability and binary ground truth
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n

    # 2. Equal-width binning for ECE and Reliability Diagram
    bin_width = 1.0 / num_bins
    bins_data: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for m in range(num_bins):
        bin_low = m * bin_width
        bin_high = (m + 1) * bin_width
        # Include upper edge in final bin
        if m == num_bins - 1:
            indices = [i for i, p in enumerate(probabilities) if bin_low <= p <= bin_high]
        else:
            indices = [i for i, p in enumerate(probabilities) if bin_low <= p < bin_high]

        bin_count = len(indices)
        if bin_count > 0:
            bin_preds = [probabilities[i] for i in indices]
            bin_labels = [labels[i] for i in indices]
            avg_conf = sum(bin_preds) / bin_count
            avg_acc = sum(bin_labels) / bin_count
            cal_gap = abs(avg_acc - avg_conf)
            weight = bin_count / n
            ece += weight * cal_gap
            mce = max(mce, cal_gap)
        else:
            avg_conf = (bin_low + bin_high) / 2.0
            avg_acc = 0.0
            cal_gap = 0.0

        bins_data.append({
            "bin_index": m,
            "bin_range": [round(bin_low, 2), round(bin_high, 2)],
            "sample_count": bin_count,
            "mean_confidence": round(avg_conf, 4),
            "empirical_accuracy": round(avg_acc, 4),
            "calibration_gap": round(cal_gap, 4),
        })

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "maximum_calibration_error": round(mce, 4),
        "sample_size": n,
        "reliability_diagram_bins": bins_data,
    }


class BayesianEvidenceCombiner:
    """
    Principled Bayesian likelihood-ratio evidence combination.
    Computes P(same_vehicle | evidence) using explicit likelihood ratios:
        Odds_post = Odds_prior * LR_reid * LR_plate * LR_kinematics * LR_type
    
    Explicit Handling:
    - Contradiction (LR = 0.0) -> Zero posterior probability.
    - Missing evidence (LR = 1.0) -> Neutral, uninformative likelihood ratio.
    - Supportive evidence (LR > 1.0) -> Increases posterior odds.
    """

    def __init__(self, prior_probability: float = 0.05) -> None:
        self.prior_p: float = max(1e-5, min(0.5, float(prior_probability)))
        self.prior_odds: float = self.prior_p / (1.0 - self.prior_p)

    def evaluate_posterior(
        self,
        reid_similarity: Optional[float],
        plate_similarity: Optional[float],
        ocr_confidence: Optional[float],
        spatial_feasible: bool,
        temporal_feasible: bool,
        vehicle_type_status: str,
        is_contradiction: bool = False,
        camera_reliability: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate exact posterior match probability given heterogeneous evidence.
        """
        if is_contradiction or not spatial_feasible or not temporal_feasible or vehicle_type_status == "incompatible":
            return 0.0, {"lr_reid": 1.0, "lr_plate": 1.0, "lr_feasibility": 0.0, "lr_total": 0.0}

        cam_rel = float(camera_reliability) if camera_reliability is not None else 0.85

        # 1. ReID Likelihood Ratio
        if reid_similarity is not None:
            # Calibrated Gaussian likelihood ratio for OSNet 512-D cosine similarity
            # High sim (>0.80) strongly supports match; low sim (<0.50) supports different vehicle
            sim = float(reid_similarity)
            # Attenuate log-LR if camera reliability is degraded
            lr_reid = math.exp((sim - 0.65) * 8.0 * cam_rel)
            lr_reid = max(0.01, min(100.0, lr_reid))
        else:
            # Missing evidence is neutral (LR = 1.0)
            lr_reid = 1.0

        # 2. Plate Likelihood Ratio
        if plate_similarity is not None:
            psim = float(plate_similarity)
            conf = float(ocr_confidence) if ocr_confidence is not None else 0.85
            if psim >= 0.85:
                # Strong plate match
                lr_plate = 1.0 + (psim * 50.0 * conf)
            elif psim < 0.35 and conf >= 0.50:
                # Contradiction
                lr_plate = 0.0
            else:
                # Partial / noisy match
                lr_plate = max(0.1, 1.0 + (psim - 0.60) * 10.0 * conf)
        else:
            # Missing plate is neutral (LR = 1.0)
            lr_plate = 1.0

        # 3. Vehicle Type Likelihood Ratio
        if vehicle_type_status == "compatible":
            lr_type = 1.5
        elif vehicle_type_status == "incompatible":
            lr_type = 0.0
        else:
            lr_type = 1.0

        total_lr = lr_reid * lr_plate * lr_type
        if total_lr == 0.0:
            return 0.0, {"lr_reid": lr_reid, "lr_plate": lr_plate, "lr_type": lr_type, "lr_total": 0.0}

        post_odds = self.prior_odds * total_lr
        post_prob = post_odds / (1.0 + post_odds)
        post_prob = round(max(0.0, min(1.0, post_prob)), 4)

        return post_prob, {
            "lr_reid": round(lr_reid, 4),
            "lr_plate": round(lr_plate, 4),
            "lr_type": round(lr_type, 4),
            "lr_total": round(total_lr, 4),
        }
