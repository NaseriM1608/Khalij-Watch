"""Labels for our SAR detections from Global Fishing Watch's AIS-matched detections.

Global Fishing Watch detects vessels in the same Sentinel-1 scenes and matches
them to satellite and terrestrial AIS, which reaches open water that shore
receivers cannot. Its public 4Wings SAR presence report is gridded, not per
detection: on the HIGH grid each row is a 0.01 degree cell (about 1.1 km)
and an hour, positioned at the cell centre, with vessel identity on AIS-matched
rows when grouped by VESSEL_ID. (Checked against the Fujairah anchorage, where
GFW's per-cell counts equal our CFAR counts in 95 of 112 cells when the
coordinates are read as centres, against 48 of 112 when read as corners.)

A label is therefore only as good as its cell is unambiguous. One of our
detections is labelled only when its cell holds exactly one matched vessel, no
unmatched GFW detection, and no other detection of ours. Near a cell edge the
two georeferencings (ours and GFW's, tens of metres apart) could put the same
ship in different cells, so a detection there is ambiguous if GFW has anything
in the cell across that edge. An earlier flat rule, ambiguous whenever near an
edge, discarded about a third of every cell even where the neighbours were
empty.

Match rates at Fujairah were 98.7% in September 2024 and 99.5% in September
2025, but 16% in June 2026 and 53% in September 2026, so labels are drawn from
passes before 2026.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# The 4Wings HIGH spatial resolution.
CELL_DEG = 0.01
# About 100 m: detections this close to a cell edge are not labelled.
EDGE_MARGIN_DEG = 0.001
# A pass lasts under a minute; a margin catches one straddling an hour boundary.
PASS_MARGIN = dt.timedelta(minutes=2)

HOUR_FORMAT = "%Y-%m-%d %H:%M"

LABELLED = "labelled"
GFW_UNMATCHED = "gfw_unmatched"
GFW_ABSENT = "gfw_absent"
AMBIGUOUS = "ambiguous"

# GFW class names (gear_type for non-fishing vessels carries the ship class,
# as TANKER or CARGO in real responses) mapped to the project's coarse
# classes. Only clear cases are mapped; the raw values are always kept.
_COARSE_CLASSES = {
    "TANKER": "cargo_tanker",
    "CARGO": "cargo_tanker",
    "CARRIER": "cargo_tanker",
    "BUNKER": "cargo_tanker",
    "FISHING": "fishing",
    "TUG": "service_tug",
    "PILOT": "service_tug",
    "SUPPORT": "service_tug",
    "DREDGE": "service_tug",
    "PASSENGER": "passenger_other",
}


@dataclass(frozen=True)
class GfwCell:
    """One 4Wings row: detections in one 0.01 degree cell during one hour."""

    lat: float
    lon: float
    hour: dt.datetime
    detections: int
    matched: bool
    vessel_id: str | None = None
    mmsi: int | None = None
    imo: str | None = None
    ship_name: str | None = None
    flag: str | None = None
    vessel_type: str | None = None
    gear_type: str | None = None

    @property
    def key(self) -> tuple[int, int]:
        return _cell_key(self.lon, self.lat)


@dataclass(frozen=True)
class Assignment:
    """What GFW says about one of our detections.

    ``reason`` explains an ambiguous one: ``crowded_ours`` (another of our
    detections shares the cell), ``crowded_gfw`` (GFW saw more than one object
    there) or ``edge`` (GFW has something just across a nearby cell edge).
    """

    status: str
    vessel: GfwCell | None = None
    reason: str | None = None


def parse_rows(rows: Iterable[dict], *, matched: bool) -> list[GfwCell]:
    """Turn report rows into cells, dropping any that cannot be placed in space and time."""
    cells: list[GfwCell] = []
    for row in rows:
        lat, lon, detections = row.get("lat"), row.get("lon"), row.get("detections") or 0
        hour = _parse_hour(row.get("date"))
        if lat is None or lon is None or hour is None or detections <= 0:
            continue
        cells.append(
            GfwCell(
                lat=float(lat),
                lon=float(lon),
                hour=hour,
                detections=int(detections),
                matched=matched,
                vessel_id=row.get("vessel_id") or None,
                mmsi=_as_int(row.get("mmsi")),
                imo=row.get("imo") or None,
                ship_name=row.get("ship_name") or None,
                flag=row.get("flag") or None,
                vessel_type=row.get("vessel_type") or None,
                gear_type=row.get("gear_type") or None,
            )
        )
    return cells


def cells_for_pass(cells: Iterable[GfwCell], acquired: dt.datetime) -> list[GfwCell]:
    """Cells from the hour, or two hours, spanned by a pass acquired at ``acquired``."""
    hours = {_floor_hour(acquired - PASS_MARGIN), _floor_hour(acquired + PASS_MARGIN)}
    return [cell for cell in cells if cell.hour in hours]


def associate(
    cells: Iterable[GfwCell],
    positions: Sequence[tuple[float, float]],
    *,
    edge_margin_deg: float = EDGE_MARGIN_DEG,
) -> list[Assignment]:
    """What GFW says about each of our detections, given as (lon, lat)."""
    vessels: dict[tuple[int, int], dict[str, GfwCell]] = defaultdict(dict)
    unmatched: dict[tuple[int, int], int] = defaultdict(int)
    for cell in cells:
        if cell.matched:
            # Distinct vessels: overlapping frames of one pass can see a ship twice.
            vessels[cell.key][cell.vessel_id or str(cell.mmsi)] = cell
        else:
            unmatched[cell.key] += cell.detections

    keys = [_cell_key(lon, lat) for lon, lat in positions]
    ours_per_cell: dict[tuple[int, int], int] = defaultdict(int)
    for key in keys:
        ours_per_cell[key] += 1

    def gfw_saw_something(cell_key: tuple[int, int]) -> bool:
        return bool(vessels.get(cell_key)) or unmatched.get(cell_key, 0) > 0

    assignments: list[Assignment] = []
    for (lon, lat), key in zip(positions, keys, strict=True):
        matched_here = list(vessels.get(key, {}).values())
        unmatched_here = unmatched.get(key, 0)
        if ours_per_cell[key] != 1:
            assignments.append(Assignment(AMBIGUOUS, reason="crowded_ours"))
        elif any(gfw_saw_something(k) for k in _across_near_edges(lon, lat, key, edge_margin_deg)):
            assignments.append(Assignment(AMBIGUOUS, reason="edge"))
        elif len(matched_here) == 1 and unmatched_here == 0:
            assignments.append(Assignment(LABELLED, matched_here[0]))
        elif not matched_here and unmatched_here == 1:
            assignments.append(Assignment(GFW_UNMATCHED))
        elif not matched_here and unmatched_here == 0:
            assignments.append(Assignment(GFW_ABSENT))
        else:
            assignments.append(Assignment(AMBIGUOUS, reason="crowded_gfw"))
    return assignments


def coarse_class(cell: GfwCell) -> str | None:
    """The project's coarse class for a matched vessel, when GFW's class is clear."""
    for value in (cell.gear_type, cell.vessel_type):
        if value and value.upper() in _COARSE_CLASSES:
            return _COARSE_CLASSES[value.upper()]
    return None


def label_fields(assignment: Assignment) -> dict[str, Any]:
    """ChipRecord fields for a detection, given what GFW says about it.

    Anything short of a label stays ``unlabelled``, never ``negative``: GFW
    failing to match a detection says nothing about whether it is a ship, and
    unmatched vessels are exactly the ones this project exists to find.
    """
    vessel = assignment.vessel
    if assignment.status == LABELLED and vessel is not None:
        return {
            "kind": "vessel",
            "source": "gfw",
            "mmsi": vessel.mmsi,
            "vessel_type": coarse_class(vessel),
            "extra": {
                "gfw_status": LABELLED,
                "gfw_vessel_id": vessel.vessel_id,
                "imo": vessel.imo,
                "ship_name": vessel.ship_name,
                "flag": vessel.flag,
                "gfw_vessel_type": vessel.vessel_type,
                "gfw_gear_type": vessel.gear_type,
            },
        }
    extra: dict[str, Any] = {"gfw_status": assignment.status}
    if assignment.reason:
        extra["gfw_reason"] = assignment.reason
    return {
        "kind": "unlabelled",
        "source": "gfw",
        "mmsi": None,
        "vessel_type": None,
        "extra": extra,
    }


def _cell_key(lon: float, lat: float) -> tuple[int, int]:
    """Index of the cell whose centre is nearest; centres lie on the 0.01 degree grid."""
    return round(lat / CELL_DEG), round(lon / CELL_DEG)


def _across_near_edges(
    lon: float, lat: float, key: tuple[int, int], margin_deg: float
) -> list[tuple[int, int]]:
    """Cells across whichever edges of ``key`` the position lies near, diagonal included."""
    margin = margin_deg / CELL_DEG
    step_lat, step_lon = _edge_step(lat, margin), _edge_step(lon, margin)
    neighbours = []
    if step_lat:
        neighbours.append((key[0] + step_lat, key[1]))
    if step_lon:
        neighbours.append((key[0], key[1] + step_lon))
    if step_lat and step_lon:
        neighbours.append((key[0] + step_lat, key[1] + step_lon))
    return neighbours


def _edge_step(value: float, margin: float) -> int:
    """+1 or -1 towards the nearby cell edge along one axis, 0 if clear of both edges."""
    offset = value / CELL_DEG - round(value / CELL_DEG)
    if 0.5 - abs(offset) >= margin:
        return 0
    return 1 if offset > 0 else -1


def _parse_hour(text: object) -> dt.datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return dt.datetime.strptime(text, HOUR_FORMAT).replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _floor_hour(moment: dt.datetime) -> dt.datetime:
    return moment.astimezone(dt.UTC).replace(minute=0, second=0, microsecond=0)


def _as_int(value: object) -> int | None:
    try:
        return int(str(value)) if value not in (None, "") else None
    except ValueError:
        return None
