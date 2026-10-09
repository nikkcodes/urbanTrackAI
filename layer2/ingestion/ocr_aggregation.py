"""
UrbanTrack AI — OCR / ANPR String Normalization and Tracklet Aggregation.

Aggregates frame-level plate detections and EasyOCR strings into tracklet-level consensus data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from layer2.ingestion.canonical_models import ANPRData

CLEAN_ALPHANUMERIC = re.compile(r"[^A-Z0-9]")


def normalize_plate_string(raw: Optional[str]) -> Optional[str]:
    """
    Normalizes a raw plate string:
    - Strips leading/trailing whitespace
    - Converts to uppercase
    - Removes all punctuation, spaces, and hyphens
    - Rejects strings shorter than 3 characters (likely noise)
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = CLEAN_ALPHANUMERIC.sub("", raw.strip().upper())
    if len(cleaned) < 3:
        return None
    return cleaned


def aggregate_tracklet_ocr(vehicle_observations: List[Dict[str, Any]]) -> ANPRData:
    """
    Aggregates frame observations of a vehicle into tracklet-level ANPR data.
    """
    plate_detections_count = 0
    valid_readings: List[Dict[str, Any]] = []

    for obs in vehicle_observations:
        # Check plate detection
        p_bbox = obs.get("plate_bbox")
        if p_bbox is not None and len(p_bbox) == 4:
            plate_detections_count += 1

        # Check plate OCR text
        raw_num = obs.get("plate_number")
        norm_num = normalize_plate_string(raw_num)
        if norm_num is not None:
            conf = obs.get("plate_text_confidence")
            conf_val = float(conf) if conf is not None else 0.0
            valid_readings.append({"text": norm_num, "confidence": conf_val})

    has_plate_detection = plate_detections_count > 0

    if not valid_readings:
        return ANPRData(
            has_plate_detection=has_plate_detection,
            has_readable_ocr=False,
            aggregated_plate_text=None,
            ocr_confidence=None,
            ocr_readings_count=0,
            plate_detections_count=plate_detections_count,
            ocr_consensus_ratio=None,
        )

    # Group valid readings to find majority consensus
    string_counts = defaultdict(int)
    string_confidences = defaultdict(list)
    for r in valid_readings:
        txt = r["text"]
        string_counts[txt] += 1
        string_confidences[txt].append(r["confidence"])

    # Select majority string; break ties by highest total confidence
    best_text = max(
        string_counts.keys(),
        key=lambda txt: (string_counts[txt], sum(string_confidences[txt])),
    )

    majority_count = string_counts[best_text]
    total_readings = len(valid_readings)
    consensus_ratio = majority_count / total_readings
    avg_conf = sum(string_confidences[best_text]) / len(string_confidences[best_text])

    return ANPRData(
        has_plate_detection=True,
        has_readable_ocr=True,
        aggregated_plate_text=best_text,
        ocr_confidence=round(avg_conf, 4),
        ocr_readings_count=total_readings,
        plate_detections_count=plate_detections_count,
        ocr_consensus_ratio=round(consensus_ratio, 4),
    )
