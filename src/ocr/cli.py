from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ocr_pdf
from .engines import DonutEngine, OlmOCRVLMEengine, TrOCREngine


def save_txt(out_path: str, results: list[tuple[int, str]]) -> None:
    """Save (page, text) list to a txt file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for page, text in results:
            f.write(f"\n\n=== Page {page} ===\n")
            f.write(text.strip())
            f.write("\n")

    print(f"✅ Saved: {out.resolve()}")


def build_engine(engine_name: str, models_dir: str):
    """Return an engine instance based on name."""
    models = Path(models_dir)

    if engine_name == "olmocr":
        model_dir = models / "allenai__olmOCR-2-7B-1025-FP8"
        proc_dir = models / "Qwen__Qwen2.5-VL-7B-Instruct"
        return OlmOCRVLMEengine(model_dir=str(model_dir), processor_dir=str(proc_dir))

    if engine_name == "trocr":
        model_dir = models / "microsoft__trocr-base-printed"
        return TrOCREngine(model_dir=str(model_dir))

    if engine_name == "donut":
        model_dir = models / "naver-clova-ix__donut-base"
        return DonutEngine(model_dir=str(model_dir))

    raise ValueError("Unknown engine: olmocr, trocr, donut")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline PDF → OCR → TXT")
    parser.add_argument("--pdf", required=True, help="Path to a local PDF file")
    parser.add_argument("--out", default="out.txt", help="Output txt path")
    parser.add_argument("--engine", default="olmocr", choices=["olmocr", "trocr", "donut"], help="OCR engine")
    parser.add_argument("--models_dir", default="./models", help="Directory containing model snapshots")
    parser.add_argument("--page_start", type=int, default=1, help="Start page (1-indexed)")
    parser.add_argument("--page_end", type=int, required=True, help="End page (1-indexed)")
    parser.add_argument("--dim", type=int, default=1288, help="Render target longest dim")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    engine = build_engine(args.engine, args.models_dir)

    results = ocr_pdf(
        pdf_path=str(pdf_path),
        engine=engine,
        page_start=args.page_start,
        page_end=args.page_end,
        target_longest_image_dim=args.dim,
    )

    save_txt(args.out, results)


if __name__ == "__main__":
    main()
