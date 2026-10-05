"""Choosing the pass scripts/detect_window.py reads: the newest covering the point, or a named one.

A named pass lets an older pass be run again with the product model - the
2026-09-10 Fujairah pass of the console's dark-vessel candidates had only CFAR.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from scripts.detect_window import choose_product

SQUARE = [(56.0, 25.0), (57.0, 25.0), (57.0, 26.0), (56.0, 26.0), (56.0, 25.0)]
ELSEWHERE = [(50.0, 29.0), (51.0, 29.0), (51.0, 30.0), (50.0, 30.0), (50.0, 29.0)]


def product(name, day, footprint=SQUARE, cog=True, online=True, s3="/eodata/x"):
    return SimpleNamespace(
        name=name,
        acq_start=dt.datetime(2026, 9, day, 2, 6, tzinfo=dt.UTC),
        footprint=footprint,
        is_cog=cog,
        online=online,
        s3_path=s3,
    )


PRODUCTS = [
    product("S1D_OLD", 10),
    product("S1C_NEW", 27),
    product("S1C_ELSEWHERE", 29, footprint=ELSEWHERE),
    product("S1C_NOT_COG", 30, cog=False),
]


def test_the_newest_readable_pass_covering_the_point_is_taken():
    assert choose_product(PRODUCTS, 56.5, 25.5).name == "S1C_NEW"


def test_a_named_pass_is_taken_when_it_covers_the_point():
    assert choose_product(PRODUCTS, 56.5, 25.5, scene="S1D_OLD").name == "S1D_OLD"


def test_a_named_pass_that_does_not_cover_the_point_is_not_taken():
    assert choose_product(PRODUCTS, 56.5, 25.5, scene="S1C_ELSEWHERE") is None
    assert choose_product(PRODUCTS, 56.5, 25.5, scene="S1C_MISSING") is None


def test_nothing_covering_the_point_gives_none():
    assert choose_product(PRODUCTS, 10.0, 10.0) is None
