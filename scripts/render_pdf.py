"""Render PDF pages to PNG, to check the typeset thesis by eye.

Uses pypdfium2 (Google's PDFium) from .venv-docs
(training/requirements-docs.txt). Run:
    .venv-docs/Scripts/python scripts/render_pdf.py reports/thesis/main.pdf --out <folder> [--pages 1-3] [--dpi 110]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def page_numbers(spec: str | None, count: int) -> list[int]:
    """1-based page numbers from "3", "1-4" or "1,5-6"; all pages if None."""
    if not spec:
        return list(range(1, count + 1))
    pages: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        pages += range(int(first), int(last or first) + 1)
    return [p for p in pages if 1 <= p <= count]


def main() -> int:
    import pypdfium2 as pdfium

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages")
    parser.add_argument("--dpi", type=float, default=110.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(args.pdf))
    for number in page_numbers(args.pages, len(document)):
        image = document[number - 1].render(scale=args.dpi / 72.0).to_pil()
        image.save(args.out / f"{args.pdf.stem}-{number:03d}.png")
        print(args.out / f"{args.pdf.stem}-{number:03d}.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
