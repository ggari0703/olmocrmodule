from __future__ import annotations

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .base import OCREngine


class TrOCREngine(OCREngine):
    """Hugging Face TrOCR engine."""

    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = TrOCRProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def ocr_image(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values=pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
