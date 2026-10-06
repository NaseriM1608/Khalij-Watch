"""Tests for the thesis report builder's formatting (Persian LaTeX tables and macros)."""

from __future__ import annotations

import pytest

from scripts import make_reports as rep


def test_numbers_keep_their_order_in_rtl_text_with_persian_separators():
    r"""XePersian draws the digits Persian; \LRE keeps 0.817 from reading as 817.0."""
    assert rep.fa_num(0.8173, 3) == r"\LRE{0٫817}"
    assert rep.fa_num(112917) == r"\LRE{112٬917}"
    assert rep.fa_num(92) == r"\LRE{92}"


def test_differences_carry_their_sign():
    assert rep.fa_signed(0.0153, 3) == r"\LRE{+0٫015}"
    assert rep.fa_signed(-0.0062, 3) == r"\LRE{−0٫006}"
    assert rep.fa_signed(-0.0002, 3) == r"\LRE{0٫000}"


def test_an_interval_reads_from_low_to_high_without_brackets():
    r"""Brackets inside \LRE are mirrored by the RTL font, so the interval is spelt out."""
    assert rep.fa_interval(0.002, 0.0301, 3) == r"\LRE{+0٫002} تا \LRE{+0٫030}"
    assert rep.fa_interval(-0.0012, 0.0134, 3) == r"\LRE{−0٫001} تا \LRE{+0٫013}"


def test_latin_terms_are_wrapped_for_xepersian():
    assert rep.lr("Sentinel-1C/1D") == r"\lr{Sentinel-1C/1D}"
    assert rep.lr("50% & more") == r"\lr{50\% \& more}"


def test_a_table_is_a_booktabs_table_with_its_caption_and_label():
    text = rep.latex_table(
        ["آشکارساز", "F1"],
        [["CFAR", "0.808"], ["مدل", "0.817"]],
        caption="آزمون",
        label="tab:test",
    )
    assert text.startswith("\\begin{table}")
    expected = "\\toprule\nآشکارساز & F1 \\\\\n\\midrule\nCFAR & 0.808 \\\\\nمدل & 0.817 \\\\\n\\bottomrule"
    assert expected in text
    assert "\\caption{آزمون}" in text and "\\label{tab:test}" in text
    assert text.rstrip().endswith("\\end{table}")


def test_a_macro_file_defines_each_number_once():
    text = rep.macros({"testFOneModel": r"\LRE{0٫817}", "testFOneCfar": r"\LRE{0٫808}"})
    assert r"\newcommand{\testFOneModel}{\LRE{0٫817}}" in text
    assert text.count("\\newcommand") == 2


def test_a_macro_name_must_be_letters_only_as_tex_requires():
    with pytest.raises(ValueError, match="genS1A"):
        rep.macros({"genS1A2026": "x"})
