from __future__ import annotations
import base64
from io import BytesIO
from typing import List, Tuple
from PIL import Image

from olmocr.data.renderpdf import render_pdf_to_base64png
from .engines import OCREngine

def render_page_to_pil(pdf_path: str, page: int, target_longest_image_dim: int = 1288) -> Image.Image:
    b64 = render_pdf_to_base64png(pdf_path, page, target_longest_image_dim=target_longest_image_dim)
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

def ocr_pdf(
    pdf_path: str,
    engine: OCREngine,
    page_start: int = 1,
    page_end: int | None = None,
    target_longest_image_dim: int = 1288,
) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number, text).
    page numbers are 1-indexed to match olmocr renderer.
    """
    results: List[Tuple[int, str]] = []

    # page_end를 모르면 일단 사용자가 넣게 하는 게 안전(렌더러마다 총페이지 얻는 법이 다름)
    if page_end is None:
        raise ValueError("page_end is required (set last page number).")

    for p in range(page_start, page_end + 1):
        img = render_page_to_pil(pdf_path, p, target_longest_image_dim=target_longest_image_dim)
        text = engine.ocr_image(img)
        results.append((p, text))

    return results
