from __future__ import annotations
from abc import ABC, abstractmethod
from PIL import Image

class OCREngine(ABC):
    """Model-agnostic OCR engine interface."""

    @abstractmethod
    def ocr_image(self, image: Image.Image) -> str:
        """Return extracted text from a single image."""
        raise NotImplementedError
