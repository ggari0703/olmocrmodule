from __future__ import annotations

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from .base import OCREngine


class DonutEngine(OCREngine):
    """Donut OCR engine."""

    def __init__(
        self,
        model_dir: str,
        device: str | None = None,
        task_prompt: str = "<s_donut><ocr>",
    ):
        self.task_prompt = task_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = DonutProcessor.from_pretrained(
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
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=512,
        )

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return text.strip()
