"""Spatial indexing and bounding-box helpers.

SQLite has no geometry type, but its built-in R*Tree module gives a genuine
spatial index. Each spatially-queried table gets an R*Tree companion kept in
sync by triggers, so the index cannot drift from the data.

On PostGIS these companions disappear and are replaced by GiST indexes on real
geometry columns. Callers work through :func:`bbox_filter_sql`, so the swap is
confined to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine, text

# Tables indexed spatially, mapped to the columns holding their extent.
# Point tables repeat the same column for min and max.
_POINT_TABLES = {
    "ais_positions": ("lon", "lat"),
    "detections": ("lon", "lat"),
    "events": ("lon", "lat"),
}
_BBOX_TABLES = {
    "sar_scenes": ("min_lon", "max_lon", "min_lat", "max_lat"),
}


@dataclass(frozen=True)
class BBox:
    """An axis-aligned WGS84 bounding box, in degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if self.min_lon > self.max_lon:
            raise ValueError(f"min_lon {self.min_lon} exceeds max_lon {self.max_lon}")
        if self.min_lat > self.max_lat:
            raise ValueError(f"min_lat {self.min_lat} exceeds max_lat {self.max_lat}")
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("longitude out of range [-180, 180]")
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("latitude out of range [-90, 90]")

    @classmethod
    def parse(cls, spec: str) -> BBox:
        """Parse ``"min_lon,min_lat,max_lon,max_lat"`` as used in .env."""
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"bbox needs 4 comma-separated values, got {len(parts)}: {spec!r}")
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"bbox contains a non-numeric value: {spec!r}") from exc
        return cls(values[0], values[1], values[2], values[3])

    def contains(self, lon: float, lat: float) -> bool:
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat

    def as_aisstream(self) -> list[list[list[float]]]:
        """aisstream.io wants [[[lat, lon], [lat, lon]]] - note lat FIRST."""
        return [[[self.min_lat, self.min_lon], [self.max_lat, self.max_lon]]]

    def as_wkt(self) -> str:
        return (
            f"POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}))"
        )


def _point_rtree_ddl(table: str, lon_col: str, lat_col: str) -> list[str]:
    rtree = f"{table}_rtree"
    return [
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {rtree} "
        f"USING rtree(id, min_lon, max_lon, min_lat, max_lat)",
        # A point is a degenerate box, so min and max are the same column.
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_ai AFTER INSERT ON {table}
            WHEN NEW.{lon_col} IS NOT NULL AND NEW.{lat_col} IS NOT NULL BEGIN
              INSERT OR REPLACE INTO {rtree}(id, min_lon, max_lon, min_lat, max_lat)
              VALUES (NEW.id, NEW.{lon_col}, NEW.{lon_col}, NEW.{lat_col}, NEW.{lat_col});
            END""",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_au AFTER UPDATE ON {table}
            WHEN NEW.{lon_col} IS NOT NULL AND NEW.{lat_col} IS NOT NULL BEGIN
              INSERT OR REPLACE INTO {rtree}(id, min_lon, max_lon, min_lat, max_lat)
              VALUES (NEW.id, NEW.{lon_col}, NEW.{lon_col}, NEW.{lat_col}, NEW.{lat_col});
            END""",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_ad AFTER DELETE ON {table} BEGIN
              DELETE FROM {rtree} WHERE id = OLD.id;
            END""",
    ]


def _bbox_rtree_ddl(table: str, cols: tuple[str, str, str, str]) -> list[str]:
    rtree = f"{table}_rtree"
    min_lon, max_lon, min_lat, max_lat = cols
    values = f"NEW.id, NEW.{min_lon}, NEW.{max_lon}, NEW.{min_lat}, NEW.{max_lat}"
    return [
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {rtree} "
        f"USING rtree(id, min_lon, max_lon, min_lat, max_lat)",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_ai AFTER INSERT ON {table} BEGIN
              INSERT OR REPLACE INTO {rtree}(id, min_lon, max_lon, min_lat, max_lat)
              VALUES ({values});
            END""",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_au AFTER UPDATE ON {table} BEGIN
              INSERT OR REPLACE INTO {rtree}(id, min_lon, max_lon, min_lat, max_lat)
              VALUES ({values});
            END""",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_rtree_ad AFTER DELETE ON {table} BEGIN
              DELETE FROM {rtree} WHERE id = OLD.id;
            END""",
    ]


def create_rtree_tables(engine: Engine) -> None:
    """Create R*Tree companions and their sync triggers. SQLite only."""
    if engine.dialect.name != "sqlite":
        return
    statements: list[str] = []
    for table, (lon_col, lat_col) in _POINT_TABLES.items():
        statements.extend(_point_rtree_ddl(table, lon_col, lat_col))
    for table, cols in _BBOX_TABLES.items():
        statements.extend(_bbox_rtree_ddl(table, cols))

    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def bbox_filter_sql(table: str) -> str:
    """SQL fragment selecting ids inside a bbox, for use in ``IN (...)``.

    Binds: ``min_lon``, ``max_lon``, ``min_lat``, ``max_lat``. An R*Tree query
    is an overlap test, which for point tables is exact containment.
    """
    if table not in _POINT_TABLES and table not in _BBOX_TABLES:
        raise ValueError(f"{table!r} has no spatial index")
    return (
        f"SELECT id FROM {table}_rtree "
        "WHERE max_lon >= :min_lon AND min_lon <= :max_lon "
        "AND max_lat >= :min_lat AND min_lat <= :max_lat"
    )


def bbox_params(bbox: BBox) -> dict[str, float]:
    return {
        "min_lon": bbox.min_lon,
        "max_lon": bbox.max_lon,
        "min_lat": bbox.min_lat,
        "max_lat": bbox.max_lat,
    }


# Areas of interest, matching .env.example.
#
# The eastern edge is 58.0 deg E, not 57.0: the Strait of Hormuz sits near
# 56-57 deg E and vessel transits must be observable on BOTH sides of it, so
# the AOI has to reach into the Gulf of Oman approaches. It must also fully
# contain HORMUZ_BBOX, which is the training and evaluation area.
GULF_BBOX = BBox(47.5, 23.5, 58.0, 30.5)
HORMUZ_BBOX = BBox(54.0, 24.5, 57.5, 27.5)
