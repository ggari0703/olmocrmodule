from __future__ import annotations
import os, base64
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
from .base import OCREngine

class OlmOCRVLMEengine(OCREngine):
    def __init__(self, model_dir: str, processor_dir: str, device: str | None = None):
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, device_map="auto", local_files_only=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            processor_dir, local_files_only=True
        )

    @torch.inference_mode()
    def ocr_image(self, image: Image.Image) -> str:
        # VLM은 이미지가 base64로 들어가는 prompt 포맷이 필요(샘플 그대로)
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=512,
            do_sample=True,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_len:]
        return self.processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
