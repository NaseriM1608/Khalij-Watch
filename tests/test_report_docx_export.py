"""A failed Word export must preserve the last reviewable document/PDF pair."""

from pathlib import Path
from types import SimpleNamespace

from scripts import build_report_docx


def test_failed_word_export_preserves_existing_outputs(tmp_path, monkeypatch):
    target = tmp_path / "report.docx"
    target.write_bytes(b"previous document")
    target.with_suffix(".pdf").write_bytes(b"previous PDF")
    source = tmp_path / "unique-build.docx"
    source.write_bytes(b"new draft")
    monkeypatch.setattr(
        build_report_docx.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="failed", stderr=""),
    )
    assert build_report_docx.export_with_word(source, target) == 1
    assert target.read_bytes() == b"previous document"
    assert target.with_suffix(".pdf").read_bytes() == b"previous PDF"


def test_successful_word_export_publishes_updated_document_and_pdf(tmp_path, monkeypatch):
    target = tmp_path / "report.docx"
    source = tmp_path / "unique-build.docx"
    source.write_bytes(b"new draft")

    def export(command, **_kwargs):
        opened = Path(command[command.index("-In") + 1])
        assert opened == source
        opened.write_bytes(b"field-updated document")
        Path(command[command.index("-Pdf") + 1]).write_bytes(b"new PDF")
        return SimpleNamespace(returncode=0, stdout="pages: 1", stderr="")

    monkeypatch.setattr(build_report_docx.subprocess, "run", export)
    assert build_report_docx.export_with_word(source, target) == 0
    assert target.read_bytes() == b"field-updated document"
    assert target.with_suffix(".pdf").read_bytes() == b"new PDF"
