"""Discover Sentinel-1 scenes covering the AOI and record their metadata.

Runs without credentials: the CDSE catalogue is public, and only reading pixels
needs authentication. So the coverage picture - which parts of the Gulf were
observed by radar, and when - is available immediately.

Run:  uv run python -m services.sar_ingest.discover --days 14
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass

from core.config import Settings, load_settings
from core.db.repository import SceneRepository
from core.db.session import create_all, create_db_engine, make_session_factory, session_scope
from core.db.spatial import BBox
from core.sar.catalogue import PRODUCT_TYPE_GRDH, CatalogueError, SarProduct, search

logger = logging.getLogger("sar_discover")

DEFAULT_LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class DiscoveryReport:
    """What one discovery pass found.

    ``mean_pass_gap_days`` is the gap between passes anywhere over the AOI, NOT
    the revisit interval of a single point. A wide AOI is clipped by many
    orbits, so this figure is far smaller than the ~2-3 day revisit any given
    location actually gets.
    """

    found: int
    new: int
    offline: int
    satellites: tuple[str, ...]
    mean_pass_gap_days: float
    latest: dt.datetime | None

    def render(self) -> str:
        latest = f"{self.latest:%Y-%m-%d %H:%M} UTC" if self.latest else "none"
        return (
            f"{self.found} scenes ({self.new} new, {self.offline} offline) "
            f"from {', '.join(self.satellites) or '-'}; "
            f"mean gap between passes {self.mean_pass_gap_days:.1f} d; latest {latest}"
        )


def discover(
    bbox: BBox,
    *,
    settings: Settings | None = None,
    days: int = DEFAULT_LOOKBACK_DAYS,
    now: dt.datetime | None = None,
) -> DiscoveryReport:
    """Search the catalogue and persist what it returns."""
    now = now or dt.datetime.now(dt.UTC)
    since = now - dt.timedelta(days=days)

    products = search(bbox, since=since, product_type=PRODUCT_TYPE_GRDH)
    if not products:
        logger.warning("no scenes found over %s in the last %d days", bbox, days)

    engine = create_db_engine(settings.database_url if settings else None)
    create_all(engine)
    factory = make_session_factory(engine)

    new = 0
    with session_scope(factory) as session:
        scenes = SceneRepository(session)
        for product in products:
            # An offline product is in long-term archive: catalogue it so the
            # coverage picture is complete, but mark it as unreadable for now.
            status = "discovered" if product.online else "offline"
            if scenes.upsert(product, status=status):
                new += 1

    return DiscoveryReport(
        found=len(products),
        new=new,
        offline=sum(1 for p in products if not p.online),
        satellites=tuple(sorted({p.satellite for p in products})),
        mean_pass_gap_days=_mean_revisit(products),
        latest=max((p.acq_start for p in products), default=None),
    )


def _mean_revisit(products: list[SarProduct]) -> float:
    from core.sar.catalogue import mean_revisit_days

    return mean_revisit_days(products)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--aoi",
        choices=("gulf", "hormuz"),
        default="gulf",
        help="gulf covers the platform area; hormuz is the training AOI",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    settings = load_settings()
    bbox = settings.gulf_bbox if args.aoi == "gulf" else settings.hormuz_bbox

    try:
        report = discover(bbox, settings=settings, days=args.days)
    except CatalogueError as exc:
        logger.error("discovery failed: %s", exc)
        return 1

    print(report.render())
    print("Contains modified Copernicus Sentinel data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
