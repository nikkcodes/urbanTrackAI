"""EasyOCR-based license plate text recognition."""

import re
from typing import Any, ClassVar

import cv2
import numpy as np
import torch

from . import config


class PlateOCR:
    """Preprocess plate crops and read their text with a cached EasyOCR reader."""

    _reader: ClassVar[Any | None] = None
    _unavailable: ClassVar[bool] = False

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

    def read_plate(self, plate_crop: np.ndarray) -> dict[str, str | float]:
        """Return raw text, validated cleaned text, and OCR confidence."""
        if plate_crop.size == 0 or self.__class__._unavailable:
            return {"raw_text": "", "text": "UNKNOWN", "confidence": 0.0}

        processed = self._preprocess(plate_crop)
        results = self.__class__._reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not results:
            return {"raw_text": "", "text": "UNKNOWN", "confidence": 0.0}

        best_result = max(results, key=lambda result: float(result[2]))
        raw_text = str(best_result[1])
        confidence = float(best_result[2])
        text = self._clean_text(raw_text)
        if confidence < config.OCR_CONF_THRESHOLD or not self._is_valid_plate(text):
            text = "UNKNOWN"
        return {"raw_text": raw_text, "text": text, "confidence": confidence}

    @staticmethod
    def _preprocess(plate_crop: np.ndarray) -> np.ndarray:
        """Enhance a plate crop without changing its aspect ratio."""
        padded = cv2.copyMakeBorder(
            plate_crop,
            8,
            8,
            8,
            8,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 5, 35, 35)
        if filtered.shape[1] < 120:
            filtered = cv2.resize(
                filtered,
                None,
                fx=3.0,
                fy=3.0,
                interpolation=cv2.INTER_CUBIC,
            )
        return filtered

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


def read_plate(plate_crop: np.ndarray) -> dict[str, str | float]:
    """Read a plate crop using the shared OCR reader."""
    return _ocr.read_plate(plate_crop)
