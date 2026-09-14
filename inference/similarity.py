"""
Similarity comparison utilities for vehicle observations.
Provides independent signal calculation functions for plate similarity,
appearance embedding cosine similarity, vehicle type compatibility, temporal gap, and geographic distance.
"""

from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.observation_schema import Observation


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def plate_similarity(plate1: Optional[str], plate2: Optional[str]) -> float:
    """
    Compare two OCR plate strings and return a normalized similarity score between 0.0 and 1.0.

    Handles:
        - Exact match (1.0)
        - Small OCR errors (normalized edit distance ratio)
        - Different strings (low score near 0.0)
        - Missing plates (returns 0.0)

    Args:
        plate1: First license plate string or None.
        plate2: Second license plate string or None.

    Returns:
        float: Normalized similarity score in [0.0, 1.0].
    """
    if plate1 is None or plate2 is None:
        return 0.0

    s1 = "".join(c.upper() for c in str(plate1) if c.isalnum())
    s2 = "".join(c.upper() for c in str(plate2) if c.isalnum())

    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    max_len = max(len(s1), len(s2))
    dist = _levenshtein_distance(s1, s2)

    similarity = 1.0 - (dist / max_len)
    return max(0.0, min(1.0, similarity))


def appearance_similarity(
    emb1: Optional[List[float]], emb2: Optional[List[float]]
) -> Optional[float]:
    """
    Calculate normalized cosine similarity between two appearance embeddings.

    Handles:
        - Identical vectors (1.0)
        - Similar vectors (~0.8 - 0.99)
        - Different vectors (~0.0 - 0.3)
        - Missing or None embeddings (returns None)
        - Mismatched dimension vectors or invalid non-numerical inputs (returns None)

    Args:
        emb1: First feature embedding vector or None.
        emb2: Second feature embedding vector or None.

    Returns:
        Optional[float]: Normalized similarity value in [0.0, 1.0], or None if evidence is missing/invalid.
    """
    if emb1 is None or emb2 is None:
        return None

    if not isinstance(emb1, (list, tuple)) or not isinstance(emb2, (list, tuple)):
        return None

    if len(emb1) == 0 or len(emb2) == 0 or len(emb1) != len(emb2):
        return None

    try:
        v1 = [float(x) for x in emb1]
        v2 = [float(x) for x in emb2]
    except (ValueError, TypeError):
        return None

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    cos_sim = dot_product / (norm1 * norm2)
    # Clip cosine similarity to [0.0, 1.0] for identity similarity estimation
    return max(0.0, min(1.0, cos_sim))


def validate_and_normalize_embedding(
    embedding: Optional[List[float]],
    expected_dim: Optional[int] = None,
) -> Optional[List[float]]:
    """
    Validate and L2-normalize a vehicle appearance feature vector (e.g. OSNet embedding).

    Handles:
        - Rejection of None, empty, non-sequence types
        - Enforcement of expected dimension if specified (e.g. 512 for OSNet)
        - Rejection of NaN, Infinity, or non-numeric elements
        - Rejection of all-zero vectors
        - Unit L2 normalization: v / ||v||_2

    Returns:
        Optional[List[float]]: L2-normalized vector or None if invalid.
    """
    if embedding is None or not isinstance(embedding, (list, tuple)):
        return None

    if len(embedding) == 0:
        return None

    if expected_dim is not None and len(embedding) != expected_dim:
        return None

    try:
        clean = [float(x) for x in embedding]
    except (ValueError, TypeError):
        return None

    if any(math.isnan(x) or math.isinf(x) for x in clean):
        return None

    norm = math.sqrt(sum(x * x for x in clean))
    if norm <= 1e-12:
        return None

    return [x / norm for x in clean]


def evaluate_reid_distribution(
    same_vehicle_pairs: List[Tuple[List[float], List[float]]],
    diff_vehicle_pairs: List[Tuple[List[float], List[float]]],
    threshold_step: float = 0.02,
) -> Dict[str, Any]:
    """
    Evaluate same-vehicle vs different-vehicle appearance similarity distributions on development data.
    Finds the optimal operating threshold that maximizes F1 score on the development split.
    """
    same_scores = []
    for e1, e2 in same_vehicle_pairs:
        s = appearance_similarity(e1, e2)
        if s is not None:
            same_scores.append(s)

    diff_scores = []
    for e1, e2 in diff_vehicle_pairs:
        s = appearance_similarity(e1, e2)
        if s is not None:
            diff_scores.append(s)

    def stats(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        n = len(scores)
        mean = sum(scores) / n
        var = sum((x - mean) ** 2 for x in scores) / n
        return {
            "count": n,
            "mean": round(mean, 4),
            "std": round(math.sqrt(var), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    same_stats = stats(same_scores)
    diff_stats = stats(diff_scores)

    optimal_candidates = []
    best_f1 = -1.0
    t = 0.10
    while t <= 0.95:
        tp = sum(1 for s in same_scores if s >= t)
        fn = sum(1 for s in same_scores if s < t)
        fp = sum(1 for s in diff_scores if s >= t)
        tn = sum(1 for s in diff_scores if s < t)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics = {
            "threshold": round(t, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            optimal_candidates = [metrics]
        elif abs(f1 - best_f1) <= 1e-6:
            optimal_candidates.append(metrics)

        t += threshold_step

    # Select the midpoint of the optimal plateau to maximize margin from both error boundaries
    best_metrics = optimal_candidates[len(optimal_candidates) // 2] if optimal_candidates else {}
    best_threshold = best_metrics.get("threshold", 0.70)

    return {
        "same_vehicle_distribution": same_stats,
        "different_vehicle_distribution": diff_stats,
        "operating_threshold": best_threshold,
        "optimal_metrics": best_metrics,
        "distribution_separation": round(same_stats["mean"] - diff_stats["mean"], 4),
    }


def vehicle_type_compatibility(
    type1: Optional[str], type2: Optional[str]
) -> Tuple[float, str]:
    """
    Compare two vehicle types for categorical compatibility.

    Examples:
        - 'car' vs 'car' -> (1.0, 'compatible')
        - 'bus' vs 'bus' -> (1.0, 'compatible')
        - 'car' vs 'bus' -> (0.0, 'incompatible')
        - Missing / None -> (0.5, 'unknown') (neutral evidence)

    Args:
        type1: Vehicle type of first observation or None.
        type2: Vehicle type of second observation or None.

    Returns:
        Tuple[float, str]: (compatibility_score, status_label)
    """
    if not type1 or not type2:
        return 0.5, "unknown"

    t1 = str(type1).strip().lower()
    t2 = str(type2).strip().lower()

    if t1 == t2:
        return 1.0, "compatible"

    # Hierarchy mapping / synonyms if needed
    synonyms = {
        "auto": "rickshaw",
        "suv": "car",
        "sedan": "car",
        "hatchback": "car",
        "van": "car",
    }
    norm1 = synonyms.get(t1, t1)
    norm2 = synonyms.get(t2, t2)

    if norm1 == norm2:
        return 1.0, "compatible"

    return 0.0, "incompatible"


def time_difference(
    t1: Union[datetime, Observation, float, int],
    t2: Union[datetime, Observation, float, int],
) -> float:
    """
    Calculate the time difference (t2 - t1) between two timestamps in seconds.

    Args:
        t1: First timestamp (datetime, Observation instance, or epoch/seconds float).
        t2: Second timestamp (datetime, Observation instance, or epoch/seconds float).

    Returns:
        float: Signed or absolute time difference in seconds.
    """
    if isinstance(t1, Observation):
        sec1 = t1.timestamp_seconds
    elif isinstance(t1, datetime):
        sec1 = t1.timestamp()
    elif isinstance(t1, (int, float)):
        sec1 = float(t1)
    else:
        raise TypeError(f"Invalid timestamp type for t1: {type(t1).__name__}")

    if isinstance(t2, Observation):
        sec2 = t2.timestamp_seconds
    elif isinstance(t2, datetime):
        sec2 = t2.timestamp()
    elif isinstance(t2, (int, float)):
        sec2 = float(t2)
    else:
        raise TypeError(f"Invalid timestamp type for t2: {type(t2).__name__}")

    return sec2 - sec1


def geographic_distance(
    lat1: Union[float, Observation],
    lon1: Union[float, Observation] = None,
    lat2: Optional[float] = None,
    lon2: Optional[float] = None,
) -> float:
    """
    Calculate the great-circle distance between two geographic coordinates using the Haversine formula.

    Can be called either as:
        geographic_distance(obs1, obs2)
    or:
        geographic_distance(lat1, lon1, lat2, lon2)

    Args:
        lat1: Latitude of point 1 OR first Observation instance.
        lon1: Longitude of point 1 OR second Observation instance.
        lat2: Latitude of point 2 (if coordinates passed directly).
        lon2: Longitude of point 2 (if coordinates passed directly).

    Returns:
        float: Approximate distance in meters.
    """
    if isinstance(lat1, Observation) and isinstance(lon1, Observation):
        obs1 = lat1
        obs2 = lon1
        if obs1.latitude is None or obs1.longitude is None or obs2.latitude is None or obs2.longitude is None:
            raise ValueError("Observation missing geographic coordinates (latitude/longitude is None).")
        p_lat1, p_lon1 = obs1.latitude, obs1.longitude
        p_lat2, p_lon2 = obs2.latitude, obs2.longitude
    elif (
        isinstance(lat1, (int, float))
        and isinstance(lon1, (int, float))
        and isinstance(lat2, (int, float))
        and isinstance(lon2, (int, float))
    ):
        p_lat1, p_lon1 = float(lat1), float(lon1)
        p_lat2, p_lon2 = float(lat2), float(lon2)
    else:
        raise TypeError(
            "Invalid argument types. Pass two Observation instances with coordinates or four float coordinates."
        )

    # Validate coordinate boundaries
    if not (-90.0 <= p_lat1 <= 90.0 and -90.0 <= p_lat2 <= 90.0):
        raise ValueError("Latitude values must be between -90.0 and 90.0.")
    if not (-180.0 <= p_lon1 <= 180.0 and -180.0 <= p_lon2 <= 180.0):
        raise ValueError("Longitude values must be between -180.0 and 180.0.")

    # Earth radius in meters
    R = 6371000.0

    phi1 = math.radians(p_lat1)
    phi2 = math.radians(p_lat2)
    delta_phi = math.radians(p_lat2 - p_lat1)
    delta_lambda = math.radians(p_lon2 - p_lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c


def evaluate_reid_only_baseline(
    observations: List[Observation],
    ground_truth_clusters: Dict[str, List[str]],
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluate pure Re-ID baseline (appearance cosine similarity ONLY) on observations.
    Explicitly excludes plate, temporal, spatial, vehicle type, and camera reliability evidence.

    Args:
        observations: List of Observation instances carrying appearance_embedding vectors.
        ground_truth_clusters: Ground truth mapping {vehicle_id: [obs_id1, obs_id2, ...]}.
        threshold: Operating cosine similarity threshold for positive match.

    Returns:
        Dict[str, Any]: Standard evaluation dictionary reporting precision, recall, F1,
                        false merge rate, false split rate, cluster purity, and pairwise counts.
    """
    from collections import Counter
    obs_map = {o.observation_id: o for o in observations}
    all_obs_ids = sorted(obs_map.keys())
    n = len(all_obs_ids)

    # Build ground-truth pairs
    gt_same_pairs: Set[Tuple[str, str]] = set()
    for v_id, member_ids in ground_truth_clusters.items():
        valid_members = [m for m in member_ids if m in obs_map]
        for i in range(len(valid_members)):
            for j in range(i + 1, len(valid_members)):
                gt_same_pairs.add((min(valid_members[i], valid_members[j]), max(valid_members[i], valid_members[j])))

    all_pairs: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((all_obs_ids[i], all_obs_ids[j]))

    gt_diff_pairs = set(all_pairs) - gt_same_pairs

    # Pure Re-ID evaluation: ONLY appearance_similarity, zero other signals
    predicted_edges: Set[Tuple[str, str]] = set()

    for u, v in all_pairs:
        oa, ob = obs_map[u], obs_map[v]
        if oa.appearance_embedding and ob.appearance_embedding:
            sim = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding)
            if sim is not None and sim >= threshold:
                predicted_edges.add((u, v))

    tp = len(predicted_edges & gt_same_pairs)
    fp = len(predicted_edges - gt_same_pairs)
    fn = len(gt_same_pairs - predicted_edges)
    tn = len(gt_diff_pairs - predicted_edges)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    false_merge_rate = (fp / len(gt_diff_pairs)) if gt_diff_pairs else 0.0
    false_split_rate = (fn / len(gt_same_pairs)) if gt_same_pairs else 0.0

    # Connected components clustering for purity
    parent = {oid: oid for oid in all_obs_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in predicted_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    pred_clusters: Dict[str, Set[str]] = {}
    for oid in all_obs_ids:
        r = find(oid)
        if r not in pred_clusters:
            pred_clusters[r] = set()
        pred_clusters[r].add(oid)

    gt_mapping = {}
    for v_id, o_ids in ground_truth_clusters.items():
        for oid in o_ids:
            gt_mapping[oid] = v_id

    purity_sum = 0
    for r, members in pred_clusters.items():
        class_counts = Counter(gt_mapping.get(m, "unknown") for m in members)
        purity_sum += class_counts.most_common(1)[0][1]

    cluster_purity = (purity_sum / n) if n > 0 else 1.0

    return {
        "model": "osnet_x0_25_msmt17",
        "threshold": threshold,
        "observations_count": n,
        "total_pairs_evaluated": len(all_pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merge_rate": round(false_merge_rate, 4),
        "false_split_rate": round(false_split_rate, 4),
        "cluster_purity": round(cluster_purity, 4),
        "clusters_formed_count": len(pred_clusters),
        "predicted_edges_count": len(predicted_edges),
    }
