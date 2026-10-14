"""Behavioral checks for the defense guides' compact, readable pagination."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile

import pytest

from scripts import build_defense_guides
from scripts.build_defense_guides import format_study_body
from scripts.report_docx import markup
from scripts.report_docx.render import Renderer

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def rendered(source: str) -> str:
    return Renderer(numbers={}, tables={}, charts={}, references={}).render(markup.parse(source))


def parsed(xml: str) -> ET.Element:
    return ET.fromstring(
        '<root xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        + xml
        + "</root>"
    )


def test_study_guide_text_starts_below_title_without_university_chapter_gap():
    body = format_study_body(rendered("# راهنمای مطالعه\n\n## مرحله نخست\n\nتوضیح مرحله.\n"))
    root = parsed(body)
    first = root.find(W + "p")
    assert first.find(f"{W}pPr/{W}pStyle").get(W + "val") == "StudyHeading1"
    assert int(first.find(f"{W}pPr/{W}spacing").get(W + "after")) <= 240
    assert root.findall(W + "p")[-1].find(f"{W}pPr/{W}pStyle").get(W + "val") == "StudyBody"


def test_long_parameter_table_can_continue_with_repeated_header():
    rows = "\n".join(f"| مورد {i} | توضیح |" for i in range(15))
    body = format_study_body(
        rendered(f"# روش\n\n:::table id=params\nپارامترها\n| نام | شرح |\n{rows}\n:::\n")
    )
    table = parsed(body).find(W + "tbl")
    assert table.find(f"{W}tr/{W}trPr/{W}tblHeader") is not None
    for row in table.findall(W + "tr")[1:]:
        assert all(p.find(f"{W}pPr/{W}keepNext") is None for p in row.iter(W + "p"))


def test_short_summary_table_stays_together():
    body = format_study_body(
        rendered("# روش\n\n:::table id=short\nخلاصه\n| نام | شرح |\n| الف | ب |\n| ج | د |\n:::\n")
    )
    table = parsed(body).find(W + "tbl")
    for row in table.findall(W + "tr")[:-1]:
        assert all(p.find(f"{W}pPr/{W}keepNext") is not None for p in row.iter(W + "p"))


def test_standalone_startup_command_preserves_left_to_right_order_and_digits():
    command = 'uv run --offline --with "uvicorn[standard]" python -m uvicorn services.api.main:app --port 8000'
    root = parsed(format_study_body(rendered(f"# راهنما\n\n{command}\n")))
    paragraph = root.findall(W + "p")[-1]
    assert paragraph.find(f"{W}pPr/{W}bidi").get(W + "val") == "0"
    assert paragraph.find(f"{W}pPr/{W}jc").get(W + "val") == "left"
    assert "".join(t.text or "" for t in paragraph.iter(W + "t")) == command
    assert all(run.find(f"{W}rPr/{W}rtl") is None for run in paragraph.findall(W + "r"))


@pytest.mark.parametrize("guide", ["phases_professor", "presentation"])
def test_separate_export_preserves_existing_guides(tmp_path, monkeypatch, guide):
    build = tmp_path / "build"
    build.mkdir()
    preserved = {
        f"{name}.pdf": f"existing {name}".encode()
        for name in ("phases", "technical", "phases_professor")
        if name != guide
    }
    for name, payload in preserved.items():
        (build / name).write_bytes(payload)
    exported = []

    def assemble(name, destination):
        assert name == guide
        destination.write_bytes(b"new guide source")
        return {"guide": name}

    def export(source, destination):
        exported.append(destination.name)
        destination.with_suffix(".pdf").write_bytes(source.read_bytes())
        return 0

    monkeypatch.setattr(build_defense_guides, "GUIDES", tmp_path)
    monkeypatch.setattr(build_defense_guides, "assemble_guide", assemble)
    monkeypatch.setattr(build_defense_guides, "export_with_word", export)
    monkeypatch.setattr("sys.argv", ["build_defense_guides.py", "--only", guide])
    assert build_defense_guides.main() == 0
    assert exported == [f"{guide}.docx"]
    assert (build / f"{guide}.pdf").read_bytes() == b"new guide source"
    for name, payload in preserved.items():
        assert (build / name).read_bytes() == payload


def test_professor_document_uses_its_own_contents_title_and_omits_project_name(
    tmp_path, monkeypatch
):
    metadata = {
        "author": "Author",
        "date": "2026",
        "contents_title": "Study contents",
        "guides": {
            "phases_professor": {
                "title": "Implementation phases",
                "subtitle": "Professor copy",
                "description": "Completed work",
                "contents_title": "Professor contents",
                "project_name": "",
            }
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "phases_professor.md").write_text("#* Implementation\n\nCompleted work.\n")
    monkeypatch.setattr(build_defense_guides, "GUIDES", tmp_path)
    destination = tmp_path / "professor.docx"
    build_defense_guides.assemble_guide("phases_professor", destination)
    with zipfile.ZipFile(destination) as package:
        document = ET.fromstring(package.read("word/document.xml"))
    paragraphs = ["".join(p.itertext()) for p in document.iter(W + "p")]
    assert "Professor contents" in paragraphs
    assert "Study contents" not in paragraphs
    assert "khalij-watch" not in " ".join(paragraphs)
