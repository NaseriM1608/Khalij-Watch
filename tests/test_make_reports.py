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
        short="کوتاه",
        label="tab:test",
    )
    assert text.startswith("\\begin{table}")
    expected = "\\toprule\nآشکارساز & F1 \\\\\n\\midrule\nCFAR & 0.808 \\\\\nمدل & 0.817 \\\\\n\\bottomrule"
    assert expected in text
    assert "\\label{tab:test}" in text
    assert text.rstrip().endswith("\\end{table}")


def test_the_list_of_tables_gets_the_short_caption():
    text = rep.latex_table([], [], caption="آزمون طولانی.", short="کوتاه", label="tab:t")
    assert "\\caption[کوتاه]{آزمون طولانی.}" in text


def test_a_table_wider_than_the_text_is_scaled_down_to_fit():
    text = rep.latex_table(["a"], [["b"]], caption="c", short="s", label="tab:t")
    lines = text.splitlines()
    begin = lines.index("\\begin{adjustbox}{max width=\\textwidth}")
    assert lines[begin + 1].startswith("\\begin{tabular}")
    assert lines[lines.index("\\end{tabular}") + 1] == "\\end{adjustbox}"


def test_a_macro_file_defines_each_number_once():
    text = rep.macros({"testFOneModel": r"\LRE{0٫817}", "testFOneCfar": r"\LRE{0٫808}"})
    assert r"\newcommand{\testFOneModel}{\LRE{0٫817}}" in text
    assert text.count("\\newcommand") == 2


def test_a_macro_name_must_be_letters_only_as_tex_requires():
    with pytest.raises(ValueError, match="genS1A"):
        rep.macros({"genS1A2026": "x"})


def test_a_coast_line_skips_tile_cut_edges_and_thins_points():
    # Land cut at the 1 degree tile edges lon 51 and lat 25; its real coast runs
    # from (51, 24.5) to (52, 25) through closely spaced points.
    coast = [(51.0 + i / 100, 24.5 + i / 200) for i in range(101)]
    ring = [(51.0, 25.0), (51.0, 24.5), *coast[1:], (52.0, 25.0), (51.0, 25.0)]
    rows = rep.coast_line(ring, spacing_deg=0.1)
    segments = [[]]
    for lon, lat in rows:
        if lon == "nan":
            segments.append([])
        else:
            segments[-1].append((lon, lat))
    drawn = [s for s in segments if len(s) > 1]
    assert len(drawn) == 1  # only the real coast is drawn
    assert drawn[0][0] == (51.0, 24.5) and drawn[0][-1] == (52.0, 25.0)
    assert len(drawn[0]) < 20  # thinned from 101 points


def test_a_coast_line_skips_tile_edges_that_land_was_buffered_past():
    # OSM land tiles overlap their neighbours by ~0.0005 degrees, so the cut edges
    # sit just off the whole-degree lines; a fully inland tile is only cut edges.
    inland = [
        (56.9994, 29.9995),
        (58.0006, 29.9995),
        (58.0006, 31.0005),
        (56.9994, 31.0005),
        (56.9994, 29.9995),
    ]
    assert rep.coast_line(inland) == []


def _config(threshold: float, same: float, neighbour: float, unmatched: float) -> dict:
    return {
        "source_label": "e05_t256o64",
        "tile": 256,
        "overlap": 64,
        "threshold": threshold,
        "f1_same_cell": same,
        "f1_neighbour_cells": neighbour,
        "recall_unmatched_neighbour": unmatched,
    }


def test_the_validation_table_marks_the_chosen_setting_not_the_top_row():
    top = _config(0.65, 0.831, 0.865, 0.798)
    chosen = _config(0.6, 0.830, 0.862, 0.868)
    text = rep.validation_table({"choice": chosen, "top10": [top, chosen]})
    rows = [line for line in text.splitlines() if "\\LRE{256}" in line]
    assert len(rows) == 2
    assert "\\textbf" not in rows[0]
    assert "\\textbf{\\LRE{0٫60}}" in rows[1]
    assert "سطر نخست" not in text
    assert r"بازیابی بدون \lr{AIS} (همسایه)" in text  # the neighbour-cell rule
