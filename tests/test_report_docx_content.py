"""Tests for the report's metadata and reference-list files."""

from __future__ import annotations

import pytest

from scripts.report_docx import content


def test_metadata_has_fields_long_blocks_and_abbreviations():
    src = (
        "title_fa: آشکارسازی شناورها\n"
        "author_en: Mahmoud Naseri\n"
        ":::abstract_en\nFirst paragraph.\n\nSecond paragraph.\n:::\n"
        ":::abbreviations\nAIS | سامانه شناسایی خودکار\nSAR | رادار روزنه مصنوعی\n:::\n"
    )
    meta = content.parse_meta(src)
    assert meta["title_fa"] == "آشکارسازی شناورها"
    assert meta["author_en"] == "Mahmoud Naseri"
    assert meta["abstract_en"] == "First paragraph.\n\nSecond paragraph."
    assert meta["abbreviations"] == [
        ("AIS", "سامانه شناسایی خودکار"),
        ("SAR", "رادار روزنه مصنوعی"),
    ]


def test_references_are_key_and_formatted_text():
    src = (
        "%% comment\n"
        'torres2012 | R. Torres et al., "GMES Sentinel-1 mission," Remote Sens. Environ., vol. 120, 2012.\n'
        "\n"
        'crisp2004 | D. J. Crisp, "The state-of-the-art in ship detection," DSTO-RR-0272, 2004.\n'
    )
    refs = content.parse_references(src)
    assert list(refs) == ["torres2012", "crisp2004"]
    assert refs["crisp2004"].startswith("D. J. Crisp")


def test_a_duplicated_reference_key_fails_loudly():
    with pytest.raises(ValueError, match="torres2012"):
        content.parse_references("torres2012 | A.\ntorres2012 | B.\n")
