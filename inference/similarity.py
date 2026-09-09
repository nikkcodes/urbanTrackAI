"""
Similarity comparison utilities for vehicle observations.
Provides independent signal calculation functions for plate similarity,
appearance embedding cosine similarity, vehicle type compatibility, temporal gap, and geographic distance.
"""

from datetime import datetime
import math
from typing import List, Optional, Tuple, Union

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
