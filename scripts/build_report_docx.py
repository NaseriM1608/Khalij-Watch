"""Build the Persian project report as a Word document on the university template.

Inputs: reports/word/content/ (meta.md, NN-*.md chapters, references.md,
diagrams/*.tex), reports/word/template.docx, and the outputs of
scripts/make_reports.py (reports/tables/*.tex, reports/data/*.csv,
reports/figures/*.png). Output: reports/word/build/report.docx, and with --pdf
the same document with its fields updated by Word plus report.pdf.

Run (charts need .venv-docs for pypdfium2):
    .venv-docs/Scripts/python scripts/build_report_docx.py --charts --pdf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_docx import charts, content, markup  # noqa: E402
from scripts.report_docx.document import Assembler, Meta  # noqa: E402
from scripts.report_docx.latex_inputs import parse_numbers, parse_table  # noqa: E402
from scripts.report_docx.render import Renderer  # noqa: E402

CONTENT = ROOT / "reports" / "word" / "content"
BUILD = ROOT / "reports" / "word" / "build"
XELATEX = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/MiKTeX/miktex/bin/x64/xelatex.exe"


def load_meta(path: Path) -> Meta:
    raw = content.parse_meta(path.read_text(encoding="utf-8"))
    return Meta(**raw)


def export_with_word(source: Path, target: Path) -> int:
    """Export under a fresh name; keep the previous pair until Word succeeds."""
    pdf = source.with_suffix(".pdf")
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "word_export.ps1"),
            "-In",
            str(source),
            "-Pdf",
            str(pdf),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode == 0:
        if not pdf.is_file():
            raise FileNotFoundError(pdf)
        source.replace(target)
        pdf.replace(target.with_suffix(".pdf"))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=BUILD / "report.docx")
    parser.add_argument("--charts", action="store_true", help="rebuild the chart and diagram PNGs")
    parser.add_argument("--pdf", action="store_true", help="update fields in Word and export a PDF")
    args = parser.parse_args()

    tables_dir = ROOT / "reports" / "tables"
    numbers = parse_numbers((tables_dir / "numbers.tex").read_text(encoding="utf-8"))
    tables = {
        p.stem: parse_table(p.read_text(encoding="utf-8"))
        for p in sorted(tables_dir.glob("*.tex"))
        if p.stem != "numbers"
    }

    chart_dir = BUILD / "charts"
    diagrams = {
        p.stem: p.read_text(encoding="utf-8") for p in sorted((CONTENT / "diagrams").glob("*.tex"))
    }
    expected = set(charts.CHARTS) | set(diagrams)
    if args.charts or any(not (chart_dir / f"{n}.png").is_file() for n in expected):
        charts.build(chart_dir, (ROOT / "reports" / "data").resolve(), XELATEX, extra=diagrams)

    assembler = Assembler(ROOT / "reports" / "word" / "template.docx")
    images = {p.stem: p for p in (ROOT / "reports" / "figures").glob("*.png")}
    images |= {n: chart_dir / f"{n}.png" for n in expected}
    registered = {
        name: assembler.add_image(name, path.read_bytes()) for name, path in sorted(images.items())
    }

    references = content.parse_references((CONTENT / "references.md").read_text(encoding="utf-8"))
    blocks = []
    for path in sorted(CONTENT.glob("[0-9][0-9]-*.md")):
        blocks += markup.parse(path.read_text(encoding="utf-8"))
    renderer = Renderer(numbers=numbers, tables=tables, charts=registered, references=references)
    meta = load_meta(CONTENT / "meta.md")
    body = assembler.body(meta, blocks, renderer)
    args.out = args.out.resolve()
    source = (
        args.out.with_stem(f"{args.out.stem}-build-{uuid4().hex[:12]}") if args.pdf else args.out
    )
    assembler.write(source, body, renderer.footnotes, renderer)

    uncited = [k for k in references if k not in renderer.cited]
    print(
        f"{args.out}: {len(renderer.chapters)} chapters, {len(renderer.cited)} references cited,"
        f" {len(renderer.footnotes)} footnotes"
    )
    if uncited:
        print("not cited (left out of the list):", ", ".join(uncited))
    if args.pdf:
        return export_with_word(source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
