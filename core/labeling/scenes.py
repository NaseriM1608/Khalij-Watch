"""Choosing which Sentinel-1 scenes go into a dataset."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any

# Consecutive frames of one pass start seconds apart; passes are hours apart.
SAME_PASS = dt.timedelta(minutes=5)


def one_per_pass[P: Any](products: Iterable[P]) -> list[P]:
    """The first product of each pass, in time order.

    Adjacent frames of one pass overlap and see the same ships at the same
    moment. Keeping both would put near-identical chips of one vessel into the
    dataset twice, and possibly into two different splits.
    """
    chosen: list[P] = []
    for product in sorted(products, key=lambda p: p.acq_start):
        previous = chosen[-1] if chosen else None
        if (
            previous is not None
            and product.satellite == previous.satellite
            and product.acq_start - previous.acq_start < SAME_PASS
        ):
            continue
        chosen.append(product)
    return chosen
