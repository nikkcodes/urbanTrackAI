"""EasyOCR-based license plate text recognition."""

import json
import re
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import torch

from . import config

# Minimum plate-crop height (pixels) before the image is upscaled for OCR.
MIN_PLATE_CROP_HEIGHT = 48
# Upscale factor applied to crops shorter than MIN_PLATE_CROP_HEIGHT.
UPSCALE_FACTOR = 2.0
# Contrast threshold below which adaptive thresholding is applied.
LOW_CONTRAST_STDDEV = 18.0
# Maximum number of plate crops exported during a single pipeline run.
OCR_DEBUG_EXPORT_MAX_SAMPLES = 100


class PlateOCR:
    """Preprocess plate crops and read their text with a cached EasyOCR reader."""

    _reader: ClassVar[Any | None] = None
    _unavailable: ClassVar[bool] = False
    # Debug statistics accumulated across read_plate() calls.
    _debug_stats: ClassVar[dict[str, int]] = {
        "calls": 0,
        "upscaled": 0,
        "adaptive_thresholded": 0,
    }
    # Debug export state (reset per pipeline run via reset_debug_export()).
    _debug_export_dir: ClassVar[Path | None] = None
    _debug_export_count: ClassVar[int] = 0
    _debug_export_records: ClassVar[list[dict[str, Any]]] = []

    def __init__(self) -> None:
        """Create the shared EasyOCR reader on the best available device."""
        if self.__class__._reader is None and not self.__class__._unavailable:
            try:
                import easyocr
            except ImportError:
                self.__class__._unavailable = True
            else:
                self.__class__._reader = easyocr.Reader(
                    ["en"],
                    gpu=torch.cuda.is_available(),
                )

    @classmethod
    def reset_debug_export(cls, camera_id: str | None = None) -> None:
        """Reset debug-export state for a new pipeline run."""
        if not config.OCR_DEBUG_EXPORT:
            cls._debug_export_dir = None
            cls._debug_export_count = 0
            cls._debug_export_records = []
            return
        export_dir = Path("data/debug/ocr_samples")
        if camera_id:
            export_dir = export_dir / camera_id
        export_dir.mkdir(parents=True, exist_ok=True)
        cls._debug_export_dir = export_dir
        cls._debug_export_count = 0
        cls._debug_export_records = []

    def read_plate(
        self,
        plate_crop: np.ndarray,
        camera_id: str | None = None,
        frame_number: int | None = None,
        track_id: int | None = None,
        detector_confidence: float | None = None,
    ) -> dict[str, str | float]:
        """Return cleaned plate number and OCR confidence.

        Optional ``camera_id``/``frame_number``/``track_id``/
        ``detector_confidence`` are used only for debug export when
        ``OCR_DEBUG_EXPORT`` is enabled; they do not affect OCR behaviour.
        """
        if plate_crop.size == 0 or self.__class__._unavailable:
            return {"plate_number": "UNKNOWN", "confidence": 0.0}

        # Preserve the original crop untouched for debugging.
        original = plate_crop
        processed = self._preprocess(plate_crop)
        results = self.__class__._reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not results:
            self._export_debug_sample(
                original, processed, camera_id, frame_number, track_id,
                detector_confidence, "UNKNOWN", 0.0,
            )
            return {"plate_number": "UNKNOWN", "confidence": 0.0}

        best_result = max(results, key=lambda result: float(result[2]))
        raw_text = str(best_result[1])
        confidence = float(best_result[2])
        text = self._clean_text(raw_text)
        if confidence < config.OCR_CONF_THRESHOLD or not self._is_valid_plate(text):
            self._export_debug_sample(
                original, processed, camera_id, frame_number, track_id,
                detector_confidence, text, confidence,
            )
            return {"plate_number": text, "confidence": confidence}

        self._export_debug_sample(
            original, processed, camera_id, frame_number, track_id,
            detector_confidence, text, confidence,
        )
        return {"plate_number": text, "confidence": confidence}

    def _export_debug_sample(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        camera_id: str | None,
        frame_number: int | None,
        track_id: int | None,
        detector_confidence: float | None,
        ocr_text: str,
        ocr_confidence: float,
    ) -> None:
        """Save original + preprocessed crops and a summary record."""
        if not config.OCR_DEBUG_EXPORT or self.__class__._debug_export_dir is None:
            return
        if self.__class__._debug_export_count >= config.OCR_DEBUG_EXPORT_MAX_SAMPLES:
            return

        camera_id = camera_id or "unknown"
        frame_number = frame_number if frame_number is not None else 0
        track_id = track_id if track_id is not None else 0
        base = f"{camera_id}_{frame_number}_{track_id}"
        orig_path = self.__class__._debug_export_dir / f"{base}_orig.png"
        proc_path = self.__class__._debug_export_dir / f"{base}_proc.png"
        try:
            cv2.imwrite(str(orig_path), original)
            cv2.imwrite(str(proc_path), processed)
        except cv2.error:
            return

        self.__class__._debug_export_records.append(
            {
                "camera_id": camera_id,
                "frame_number": int(frame_number),
                "track_id": int(track_id),
                "crop_width": int(original.shape[1]),
                "crop_height": int(original.shape[0]),
                "detector_confidence": (
                    float(detector_confidence)
                    if detector_confidence is not None
                    else None
                ),
                "ocr_confidence": float(ocr_confidence),
                "ocr_text": ocr_text if ocr_text != "UNKNOWN" else None,
                "original_image": str(orig_path),
                "preprocessed_image": str(proc_path),
            }
        )
        self.__class__._debug_export_count += 1

    @classmethod
    def write_debug_summary(cls, output_path: str | Path | None = None) -> None:
        """Write the accumulated OCR debug summary to JSON."""
        if not config.OCR_DEBUG_EXPORT:
            return
        path = Path(output_path) if output_path else Path("data/debug/ocr_debug_summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "export_enabled": True,
                    "max_samples": int(config.OCR_DEBUG_EXPORT_MAX_SAMPLES),
                    "samples_exported": int(cls._debug_export_count),
                    "samples": cls._debug_export_records,
                },
                file,
                indent=4,
            )

    @staticmethod
    def _preprocess(plate_crop: np.ndarray) -> np.ndarray:
        """Enhance a plate crop without changing its aspect ratio.

        Steps applied to the cropped plate image only:
        1. Upscale crops shorter than ``MIN_PLATE_CROP_HEIGHT`` (2x).
        2. Convert to grayscale.
        3. Apply CLAHE (adaptive histogram equalization).
        4. Apply a light bilateral filter to reduce noise.
        5. Apply adaptive thresholding only when contrast is low.
        The original crop is never modified.
        """
        # 1. Upscale small crops.
        if plate_crop.shape[0] < MIN_PLATE_CROP_HEIGHT:
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=UPSCALE_FACTOR,
                fy=UPSCALE_FACTOR,
                interpolation=cv2.INTER_CUBIC,
            )
            PlateOCR._record_debug("upscaled")

        # 2. Convert to grayscale.
        if plate_crop.ndim == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop

        # 3. CLAHE (adaptive histogram equalization).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Light bilateral filter to reduce noise.
        filtered = cv2.bilateralFilter(enhanced, 5, 35, 35)

        # 5. Adaptive thresholding only for low-contrast crops.
        if float(np.std(filtered)) < LOW_CONTRAST_STDDEV:
            filtered = cv2.adaptiveThreshold(
                filtered,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                2,
            )
            PlateOCR._record_debug("adaptive_thresholded")

        return filtered

    @classmethod
    def _record_debug(cls, key: str) -> None:
        """Accumulate debug statistics in debug mode only."""
        if not config.PLATE_OCR_DEBUG:
            return
        stats = cls._debug_stats
        stats["calls"] += 1
        if key == "upscaled":
            stats["upscaled"] += 1
        elif key == "adaptive_thresholded":
            stats["adaptive_thresholded"] += 1

    @classmethod
    def debug_summary(cls) -> dict[str, int]:
        """Return accumulated OCR preprocessing statistics."""
        stats = cls._debug_stats
        return {
            "ocr_calls": int(stats["calls"]),
            "upscaled_crops": int(stats["upscaled"]),
            "adaptive_thresholded_crops": int(stats["adaptive_thresholded"]),
        }

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Normalize and correct OCR characters in numeric plate sections."""
        candidate = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if len(candidate) < 8:
            return candidate

        state = candidate[:2]
        remainder = candidate[2:]
        for numeric_length in (2, 1):
            numeric = remainder[:numeric_length]
            letters_and_number = remainder[numeric_length:]
            corrected_numeric = PlateOCR._correct_numeric_section(numeric)
            if not corrected_numeric.isdigit():
                continue
            for letter_length in range(3, 0, -1):
                letters = letters_and_number[:letter_length]
                final_number = letters_and_number[letter_length:]
                corrected_final = PlateOCR._correct_numeric_section(final_number)
                if (
                    letters.isalpha()
                    and len(corrected_final) == 4
                    and corrected_final.isdigit()
                ):
                    return f"{state}{corrected_numeric}{letters}{corrected_final}"
        return candidate

    @staticmethod
    def _correct_numeric_section(section: str) -> str:
        """Apply conservative OCR substitutions only within numeric sections."""
        corrected = section.replace("O", "0").replace("I", "1").replace("S", "5")
        corrected = re.sub(r"(?<=\d)B(?=\d)", "8", corrected)
        return corrected

    @staticmethod
    def _is_valid_plate(text: str) -> bool:
        """Accept common Indian registration formats with 1-2 digit regions."""
        return bool(re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}", text))


_ocr = PlateOCR()


def read_plate(
    plate_crop: np.ndarray,
    camera_id: str | None = None,
    frame_number: int | None = None,
    track_id: int | None = None,
    detector_confidence: float | None = None,
) -> dict[str, str | float]:
    """Read a plate crop using the shared OCR reader.

    Optional ``camera_id``/``frame_number``/``track_id``/
    ``detector_confidence`` are used only for debug export when
    ``OCR_DEBUG_EXPORT`` is enabled; they do not affect OCR behaviour.
    """
    return _ocr.read_plate(
        plate_crop,
        camera_id=camera_id,
        frame_number=frame_number,
        track_id=track_id,
        detector_confidence=detector_confidence,
    )
