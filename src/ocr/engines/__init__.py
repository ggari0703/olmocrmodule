"""Available OCR engines."""

from .base import OCREngine
from .olmocr_vlm import OlmOCRVLMEengine
from .trocr import TrOCREngine
from .donut import DonutEngine

__all__ = [
    "OCREngine",
    "OlmOCRVLMEengine",
    "TrOCREngine",
    "DonutEngine",
]
