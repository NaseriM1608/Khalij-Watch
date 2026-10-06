"""Data access. The only place that queries tables directly.

Keeping every query here is what makes the eventual PostGIS migration a
contained change rather than a rewrite: callers ask for vessels in a bounding
box, not for a particular SQL dialect's spatial predicate.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, insert, or_, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from core.ais.messages import AisPositionReport, VesselStatic, coarse_class
from core.ais.plausibility import ScoredPosition
from core.db.models import (
    AisPosition,
    AnalysedWindow,
    Detection,
    Event,
    SarScene,
    Vessel,
)
from core.db.spatial import BBox, bbox_filter_sql, bbox_params

# Positions are written in batches; a per-message round trip cannot keep up
# with a busy feed.
DEFAULT_BATCH_SIZE = 500

# SQLAlchemy stores datetimes in SQLite as naive UTC text in exactly this
# shape. Raw-SQL comparisons are therefore LEXICAL, so a bound value must match
# the format byte for byte: an ISO string ("...T06:00:00+00:00") sorts before
# every stored value because "T" > " ", silently matching nothing.
_SQLITE_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Both SQLite and PostgreSQL cap the number of bound variables in one
# statement (~32k and 65535 respectively). A multi-row INSERT binds
# columns x rows, so a large batch must be split or the driver raises
# "too many SQL variables". Chunk conservatively enough for either.
MAX_BOUND_VARIABLES = 30_000


def _chunked(rows: list[dict], table) -> list[list[dict]]:
    """Split rows so no single statement exceeds the bind-variable limit.

    The width is taken from the TABLE, not from the supplied dicts: SQLAlchemy
    renders every column for each row in a multi-row INSERT, including ones the
    caller omitted, so counting dict keys badly underestimates the real cost.
    """
    if not rows:
        return []
    width = max(len(table.columns), 1)
    per_statement = max(1, MAX_BOUND_VARIABLES // width)
    return [rows[i : i + per_statement] for i in range(0, len(rows), per_statement)]


def _as_utc(value: dt.datetime) -> dt.datetime:
    """Normalise to UTC. SQLite drops the offset, so this must happen first."""
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC)


def _bind_timestamp(session: Session, value: dt.datetime) -> object:
    """Render a timestamp for direct comparison in raw SQL."""
    utc = _as_utc(value)
    if session.get_bind().dialect.name == "sqlite":
        return utc.replace(tzinfo=None).strftime(_SQLITE_DATETIME_FORMAT)
    return utc


def _row_timestamp(value: object) -> dt.datetime:
    """Read a timestamp from a raw-SQL row.

    SQLite has no native timestamptz: values come back as naive text that is
    UTC by construction.
    """
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    return _as_utc(value)


# The point detector's name in the detections table, as written by
# core.inference.runner.DETECTOR. Repeated here so the API does not have to
# import numpy; a test keeps the two equal.
MODEL_DETECTOR = "pointdet"
CFAR_DETECTOR = "cfar"
# Which detections in_bbox returns: one detector, every row, or "auto" - per
# pass, the model's where it ran and the pass's other rows where it did not.
DETECTOR_CHOICES = ("auto", CFAR_DETECTOR, MODEL_DETECTOR, "all")

# A contact older than this is stale for live-map purposes. It is still shown,
# but the UI must render its age explicitly rather than implying it is current.
DEFAULT_MAX_AGE = dt.timedelta(hours=6)


@dataclass(frozen=True)
class VesselFix:
    """A vessel's most recent known position, for the live map."""

    mmsi: int
    lon: float
    lat: float
    ts: dt.datetime
    sog_kn: float | None
    cog_deg: float | None
    heading_deg: float | None
    name: str | None
    coarse_class: str | None
    source: str
    plausibility_score: float | None
    anomaly_flags: list | None = None

    @property
    def is_suspect(self) -> bool:
        """Below this the position contradicts itself or the geography."""
        return self.plausibility_score is not None and self.plausibility_score < 0.5

    def age_seconds(self, now: dt.datetime) -> float:
        return max(0.0, (now - self.ts).total_seconds())


class VesselRepository:
    """Vessel identities, built up from sparse AIS static messages."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_static(self, static: VesselStatic, *, seen_at: dt.datetime | None = None) -> None:
        """Merge a static record, preserving fields it does not carry.

        AIS type 24 arrives in two parts - Part A has the name, Part B the
        type, call sign and dimensions - so a naive overwrite would erase
        whichever half arrived first. Only non-None fields are applied.
        """
        values: dict[str, object] = {"mmsi": static.mmsi}
        incoming = {
            "name": static.name,
            "callsign": static.callsign,
            "imo": static.imo,
            "ship_type": static.ship_type,
            "length_m": static.length_m,
            "beam_m": static.beam_m,
            "coarse_class": coarse_class(static.ship_type),
        }
        values.update({k: v for k, v in incoming.items() if v is not None})
        if seen_at is not None:
            values["last_seen"] = seen_at

        stmt = sqlite_insert(Vessel).values(**values)
        # Only overwrite columns this message actually carried.
        updatable = {k: getattr(stmt.excluded, k) for k in values if k != "mmsi"}
        if updatable:
            stmt = stmt.on_conflict_do_update(index_elements=["mmsi"], set_=updatable)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=["mmsi"])
        self._session.execute(stmt)

    def ensure_exists(self, mmsis: set[int], *, seen_at: dt.datetime) -> None:
        """Create placeholder rows so positions never violate the foreign key.

        Positions routinely arrive before any static message for that vessel.
        """
        if not mmsis:
            return
        rows = [{"mmsi": mmsi, "first_seen": seen_at, "last_seen": seen_at} for mmsi in mmsis]
        for chunk in _chunked(rows, Vessel.__table__):
            stmt = (
                sqlite_insert(Vessel).values(chunk).on_conflict_do_nothing(index_elements=["mmsi"])
            )
            self._session.execute(stmt)

    def get(self, mmsi: int) -> Vessel | None:
        return self._session.get(Vessel, mmsi)

    def count(self) -> int:
        return self._session.execute(select(func.count()).select_from(Vessel)).scalar_one()


class PositionRepository:
    """AIS position reports."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_batch(self, reports: Sequence[AisPositionReport | ScoredPosition]) -> int:
        """Insert a batch of positions, ignoring exact duplicates.

        Accepts scored or unscored positions. The feed replays the same fix
        through multiple receivers, so duplicates are normal and must not abort
        the batch. Returns the number of rows actually written.
        """
        if not reports:
            return 0

        rows = [_position_row(entry) for entry in reports]
        written = 0
        for chunk in _chunked(rows, AisPosition.__table__):
            stmt = (
                sqlite_insert(AisPosition)
                .values(chunk)
                .on_conflict_do_nothing(index_elements=["mmsi", "ts", "lon", "lat"])
            )
            written += self._session.execute(stmt).rowcount or 0
        return written

    def count(self) -> int:
        return self._session.execute(select(func.count()).select_from(AisPosition)).scalar_one()

    def latest_in_bbox(
        self,
        bbox: BBox,
        *,
        now: dt.datetime | None = None,
        max_age: dt.timedelta | None = DEFAULT_MAX_AGE,
        limit: int = 10_000,
    ) -> list[VesselFix]:
        """Most recent fix per vessel inside a bounding box.

        This is the live map's primary query. The spatial index narrows
        candidates first; the window function then picks one row per vessel.
        """
        now = now or dt.datetime.now(dt.UTC)
        params = dict(bbox_params(bbox))
        params["row_limit"] = limit

        age_clause = ""
        if max_age is not None:
            params["cutoff"] = _bind_timestamp(self._session, now - max_age)
            age_clause = "AND p.ts >= :cutoff"

        sql = f"""
            WITH candidates AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.mmsi ORDER BY p.ts DESC
                ) AS rn
                FROM ais_positions p
                WHERE p.id IN ({bbox_filter_sql("ais_positions")})
                {age_clause}
            )
            SELECT c.mmsi, c.lon, c.lat, c.ts, c.sog_kn, c.cog_deg, c.heading_deg,
                   c.source, c.plausibility_score, c.anomaly_flags,
                   v.name, v.coarse_class
            FROM candidates c
            LEFT JOIN vessels v ON v.mmsi = c.mmsi
            WHERE c.rn = 1
            ORDER BY c.ts DESC
            LIMIT :row_limit
        """
        rows = self._session.execute(text(sql), params).mappings().all()
        return [_to_fix(row) for row in rows]

    def track(
        self,
        mmsi: int,
        *,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        limit: int = 5_000,
    ) -> list[AisPosition]:
        """One vessel's positions, oldest first, for drawing a trail."""
        stmt = select(AisPosition).where(AisPosition.mmsi == mmsi)
        if since is not None:
            stmt = stmt.where(AisPosition.ts >= since)
        if until is not None:
            stmt = stmt.where(AisPosition.ts <= until)
        stmt = stmt.order_by(AisPosition.ts).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def delete_older_than(self, cutoff: dt.datetime) -> int:
        """Retention. Positions are cheap but not free; trim the tail."""
        stmt = AisPosition.__table__.delete().where(AisPosition.ts < cutoff)
        return self._session.execute(stmt).rowcount or 0


def _position_row(entry: AisPositionReport | ScoredPosition) -> dict:
    """Flatten a position, scored or not, into a row."""
    if isinstance(entry, ScoredPosition):
        report, verdict = entry.report, entry.plausibility
        score, flags = verdict.score, verdict.as_flags()
    else:
        report, score, flags = entry, None, None
    return {
        "ts": _as_utc(report.ts),
        "mmsi": report.mmsi,
        "lon": report.lon,
        "lat": report.lat,
        "sog_kn": report.sog_kn,
        "cog_deg": report.cog_deg,
        "heading_deg": report.heading_deg,
        "nav_status": report.nav_status,
        "msg_type": report.msg_type,
        "source": report.source,
        "plausibility_score": score,
        "anomaly_flags": flags,
    }


def _to_fix(row) -> VesselFix:
    return VesselFix(
        mmsi=row["mmsi"],
        lon=row["lon"],
        lat=row["lat"],
        ts=_row_timestamp(row["ts"]),
        sog_kn=row["sog_kn"],
        cog_deg=row["cog_deg"],
        heading_deg=row["heading_deg"],
        name=row["name"],
        coarse_class=row["coarse_class"],
        source=row["source"],
        plausibility_score=row["plausibility_score"],
        anomaly_flags=_decode_flags(row["anomaly_flags"]),
    )


def _decode_flags(raw: object) -> list | None:
    """SQLite returns the JSON column as text on a raw-SQL read."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


class SceneRepository:
    """Catalogued SAR scenes.

    Only metadata is stored. Pixels are read through a window when needed and
    never persisted: a full-Gulf coverage is ~25 GB of raw imagery, and the
    project's whole storage strategy is process-and-discard.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, product, *, status: str = "discovered") -> bool:
        """Record a catalogued product. Returns True if it was new."""
        existing = self._session.execute(
            select(SarScene.id).where(SarScene.product_id == product.product_id)
        ).scalar_one_or_none()

        values = {
            "product_id": product.product_id,
            "product_name": product.name,
            "satellite": product.satellite,
            "mode": product.mode,
            "polarisation": product.polarisation,
            "acq_time": _as_utc(product.acq_start),
            "orbit_direction": product.orbit_direction,
            "min_lon": product.bbox.min_lon,
            "min_lat": product.bbox.min_lat,
            "max_lon": product.bbox.max_lon,
            "max_lat": product.bbox.max_lat,
            "footprint_wkt": _ring_to_wkt(product.footprint),
            "cog_url": product.s3_path,
            "status": status,
        }
        stmt = (
            sqlite_insert(SarScene)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["product_id"],
                set_={k: v for k, v in values.items() if k != "product_id"},
            )
        )
        self._session.execute(stmt)
        return existing is None

    def count(self) -> int:
        return self._session.execute(select(func.count()).select_from(SarScene)).scalar_one()

    def id_for(self, product_id: str) -> int | None:
        """The database id of a catalogued product, for linking its detections."""
        return self._session.execute(
            select(SarScene.id).where(SarScene.product_id == product_id)
        ).scalar_one_or_none()

    def get(self, scene_id: int) -> SarScene | None:
        """One stored scene by database id, with the object-storage path of its image."""
        return self._session.get(SarScene, scene_id)

    def intersecting(
        self,
        bbox: BBox,
        *,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        limit: int = 500,
    ) -> list[SarScene]:
        """Scenes whose footprint overlaps a bbox, acquired in [since, until], newest first."""
        params = dict(bbox_params(bbox))
        params["row_limit"] = limit
        since_clause = ""
        if since is not None:
            params["since"] = _bind_timestamp(self._session, since)
            since_clause = "AND acq_time >= :since"
        if until is not None:
            params["until"] = _bind_timestamp(self._session, until)
            since_clause += " AND acq_time <= :until"

        sql = f"""
            SELECT * FROM sar_scenes
            WHERE id IN ({bbox_filter_sql("sar_scenes")})
            {since_clause}
            ORDER BY acq_time DESC
            LIMIT :row_limit
        """
        rows = self._session.execute(text(sql), params).mappings().all()
        return [self._session.get(SarScene, row["id"]) for row in rows]

    def latest_acquisition(self, bbox: BBox) -> dt.datetime | None:
        """When this area was last observed by radar.

        Drives the coverage-honesty overlay: where this is old or absent, the
        platform is blind and must say so.
        """
        scenes = self.intersecting(bbox, limit=1)
        if not scenes:
            return None
        acq = scenes[0].acq_time
        return acq if acq.tzinfo else acq.replace(tzinfo=dt.UTC)


def _ring_to_wkt(ring: list[tuple[float, float]]) -> str | None:
    if not ring:
        return None
    points = ", ".join(f"{lon} {lat}" for lon, lat in ring)
    return f"POLYGON(({points}))"


class MeasuredTarget(Protocol):
    """What a detector must report for a detection to be stored.

    ``core.sar.detect.Candidate`` satisfies this once georeferenced. It is
    declared structurally so the API, which never runs a detector, does not
    have to import numpy.
    """

    lon: float | None
    lat: float | None
    length_m: float | None
    width_m: float | None
    heading_deg: float | None
    pixel_count: int | None
    peak_amplitude: float | None


@dataclass(frozen=True)
class DetectionView:
    """A stored detection joined to the radar pass that produced it."""

    id: int
    scene_id: int
    lon: float
    lat: float
    length_m: float | None
    width_m: float | None
    heading_deg: float | None
    confidence: float
    detector: str
    detector_version: str | None
    polarisation: str | None
    pixel_count: int | None
    peak_dn: float | None
    acq_time: dt.datetime
    satellite: str
    product_name: str
    # Metres to land (capped, scripts/detect_window.py); None if unknown.
    shore_m: float | None = None

    def age_seconds(self, now: dt.datetime) -> float:
        """Age of the observation: the age of the pass, not of the row."""
        return max(0.0, (now - self.acq_time).total_seconds())


class DetectionRepository:
    """Vessel detections found in SAR scenes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_in_bbox(
        self,
        scene_id: int,
        bbox: BBox,
        targets: Sequence[MeasuredTarget],
        *,
        polarisation: str,
        detector: str,
        detector_version: str | None,
        confidence: float,
        shore_m: Sequence[float | None] | None = None,
    ) -> int:
        """Store one window's detections, replacing any earlier run over it.

        Re-running a window is routine - after a threshold change, or by
        accident - and must not double the vessel count. Rows are replaced only
        for the same scene, channel and detector inside ``bbox``, so other
        windows and the other polarisation keep their results. Every target
        must therefore lie inside ``bbox``: one outside would survive the next
        re-run and be duplicated by it. ``shore_m`` gives each target's
        distance to land (None: unknown). Returns the number of rows written.
        """
        shared = {
            "scene_id": scene_id,
            "polarisation": polarisation,
            "detector": detector,
            "detector_version": detector_version,
            "confidence": confidence,
        }
        # Validate everything before deleting anything.
        if shore_m is not None and len(shore_m) != len(targets):
            raise ValueError(f"{len(shore_m)} distances to land for {len(targets)} detections")
        distances = shore_m if shore_m is not None else [None] * len(targets)
        rows = [
            {**_detection_row(target, bbox, shared), "shore_m": _optional(float, shore)}
            for target, shore in zip(targets, distances, strict=True)
        ]

        params: dict[str, object] = dict(bbox_params(bbox))
        params.update(scene_id=scene_id, polarisation=polarisation, detector=detector)
        # The R*Tree narrows the search, but it stores 32-bit floats rounded
        # outward and so also matches points a few decimetres outside the box.
        # Comparing the stored doubles deletes exactly what bbox.contains accepts.
        sql = f"""
            DELETE FROM detections
            WHERE scene_id = :scene_id
              AND polarisation = :polarisation
              AND detector = :detector
              AND id IN ({bbox_filter_sql("detections")})
              AND lon BETWEEN :min_lon AND :max_lon
              AND lat BETWEEN :min_lat AND :max_lat
        """
        self._session.execute(text(sql), params)
        if rows:
            self._session.execute(insert(Detection), rows)
        return len(rows)

    def in_bbox(
        self,
        bbox: BBox,
        *,
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
        limit: int = 20_000,
        detector: str = "auto",
        detector_version: str | None = None,
        include_fused: bool = False,
    ) -> list[DetectionView]:
        """Detections inside a bounding box, newest pass first.

        CFAR and the model are stored side by side, so a pass both have seen
        holds each vessel twice. ``detector`` picks what to return: "cfar" or
        "pointdet" for one detector, "all" for every row, or "auto" (the
        default) for, per pass, the model's rows where the model has rows in
        this box and the pass's other rows where it has none.
        ``detector_version`` keeps one version only (e.g. the product model's).
        ``include_fused`` keeps passes with events whatever their age, as their
        dark-vessel candidates and searched areas are kept. ``until`` leaves out
        passes acquired after it, fused or not (the console as of a past moment).
        """
        if detector not in DETECTOR_CHOICES:
            raise ValueError(f"detector must be one of {DETECTOR_CHOICES}, got {detector!r}")
        params: dict[str, object] = dict(bbox_params(bbox))
        params["row_limit"] = limit
        since_clause = ""
        if since is not None:
            params["since"] = _bind_timestamp(self._session, since)
            since_clause = "AND s.acq_time >= :since"
            if include_fused:
                since_clause = (
                    "AND (s.acq_time >= :since OR d.scene_id IN "
                    "(SELECT scene_id FROM events WHERE scene_id IS NOT NULL))"
                )
        if until is not None:
            params["until"] = _bind_timestamp(self._session, until)
            since_clause += " AND s.acq_time <= :until"
        detector_clause = ""
        if detector == "auto":
            # Since filters whole passes, the same pass is in or out for both
            # rows, so only the scene and the box need repeating here.
            params["model_detector"] = MODEL_DETECTOR
            detector_clause = f"""AND (d.detector = :model_detector OR NOT EXISTS (
                SELECT 1 FROM detections m
                WHERE m.scene_id = d.scene_id
                  AND m.detector = :model_detector
                  AND m.id IN ({bbox_filter_sql("detections")})
            ))"""
        elif detector != "all":
            params["detector"] = detector
            detector_clause = "AND d.detector = :detector"
        if detector_version is not None:
            params["detector_version"] = detector_version
            detector_clause += " AND d.detector_version = :detector_version"

        sql = f"""
            SELECT {_DETECTION_COLUMNS}
            FROM detections d
            JOIN sar_scenes s ON s.id = d.scene_id
            WHERE d.id IN ({bbox_filter_sql("detections")})
            {since_clause}
            {detector_clause}
            ORDER BY s.acq_time DESC, d.id
            LIMIT :row_limit
        """
        rows = self._session.execute(text(sql), params).mappings().all()
        return [_to_detection_view(row) for row in rows]

    def get(self, detection_id: int) -> DetectionView | None:
        """One detection by id, with its pass."""
        sql = f"""
            SELECT {_DETECTION_COLUMNS}
            FROM detections d
            JOIN sar_scenes s ON s.id = d.scene_id
            WHERE d.id = :detection_id
        """
        row = self._session.execute(text(sql), {"detection_id": detection_id}).mappings().first()
        return None if row is None else _to_detection_view(row)

    def count(self) -> int:
        return self._session.execute(select(func.count()).select_from(Detection)).scalar_one()


_DETECTION_COLUMNS = """d.id, d.scene_id, d.lon, d.lat, d.length_m, d.width_m, d.heading_deg,
                   d.confidence, d.detector, d.detector_version, d.polarisation,
                   d.pixel_count, d.peak_dn, s.acq_time, s.satellite, s.product_name,
                   d.shore_m"""


def _detection_row(target: MeasuredTarget, bbox: BBox, shared: dict[str, object]) -> dict:
    """Flatten a target into a row, refusing any that cannot be placed correctly."""
    if target.lon is None or target.lat is None:
        raise ValueError("detections must be georeferenced before they are stored")
    if not (math.isfinite(target.lon) and math.isfinite(target.lat)):
        raise ValueError(f"detection has a non-finite position: {target.lon}, {target.lat}")
    if not bbox.contains(target.lon, target.lat):
        raise ValueError(
            f"detection at {target.lat:.5f}, {target.lon:.5f} lies outside the window it "
            "replaces; clip detections to the window before storing them"
        )
    # A point detector reports a position and its own score, but no hull.
    own_confidence = getattr(target, "confidence", None)
    return {
        **shared,
        "lon": float(target.lon),
        "lat": float(target.lat),
        "length_m": _optional(float, target.length_m),
        "width_m": _optional(float, target.width_m),
        "heading_deg": _optional(float, target.heading_deg),
        "pixel_count": _optional(int, target.pixel_count),
        "peak_dn": _optional(float, target.peak_amplitude),
        **({"confidence": float(own_confidence)} if own_confidence is not None else {}),
    }


def _optional(kind: type, value: object) -> object:
    return None if value is None else kind(value)


def _to_detection_view(row) -> DetectionView:
    return DetectionView(
        id=row["id"],
        scene_id=row["scene_id"],
        lon=row["lon"],
        lat=row["lat"],
        length_m=row["length_m"],
        width_m=row["width_m"],
        heading_deg=row["heading_deg"],
        confidence=row["confidence"],
        detector=row["detector"],
        detector_version=row["detector_version"],
        polarisation=row["polarisation"],
        pixel_count=row["pixel_count"],
        peak_dn=row["peak_dn"],
        acq_time=_row_timestamp(row["acq_time"]),
        satellite=row["satellite"],
        product_name=row["product_name"],
        shore_m=row["shore_m"],
    )


@dataclass(frozen=True)
class NewEvent:
    """An event to store; ``evidence`` must explain it (PLAN §9: no unexplainable alert)."""

    event_type: str
    ts: dt.datetime
    lon: float | None
    lat: float | None
    severity: str
    detection_id: int | None
    evidence: dict


@dataclass(frozen=True)
class EventView:
    """A stored event as the API serves it."""

    id: int
    event_type: str
    ts: dt.datetime
    lon: float | None
    lat: float | None
    severity: str
    detection_id: int | None
    scene_id: int | None
    run: str | None
    evidence: dict
    acknowledged: bool


class EventRepository:
    """Analyst-facing events, each with its evidence card."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_for_scene(
        self,
        scene_id: int,
        run: str,
        events: Sequence[NewEvent],
        *,
        bbox: BBox | None = None,
    ) -> int:
        """Store one fusion run's events for a pass, replacing that run's earlier events for it.

        With ``bbox``, only the run's events of the pass inside it are replaced,
        so fusing another box of the same pass keeps the first box's events;
        every new event must then lie inside it.
        """
        for event in events:  # validate everything before deleting anything
            if not event.evidence:
                raise ValueError(f"{event.event_type} at {event.lon}, {event.lat} has no evidence")
            if bbox is not None and not bbox.contains(event.lon, event.lat):
                raise ValueError(
                    f"{event.event_type} at {event.lat:.5f}, {event.lon:.5f} lies outside "
                    "the box it replaces"
                )
        old = self._session.query(Event).filter(Event.scene_id == scene_id, Event.run == run)
        if bbox is not None:
            old = old.filter(
                Event.lon.between(bbox.min_lon, bbox.max_lon),
                Event.lat.between(bbox.min_lat, bbox.max_lat),
            )
        old.delete(synchronize_session=False)
        self._session.add_all(
            Event(
                event_type=e.event_type,
                ts=_as_utc(e.ts),
                lon=e.lon,
                lat=e.lat,
                severity=e.severity,
                detection_id=e.detection_id,
                evidence=e.evidence,
                scene_id=scene_id,
                run=run,
            )
            for e in events
        )
        self._session.flush()
        return len(events)

    def in_bbox(
        self,
        bbox: BBox,
        *,
        event_type: str | None = None,
        until: dt.datetime | None = None,
        limit: int = 5_000,
    ) -> list[EventView]:
        """Events inside a bounding box, of passes up to ``until``, newest first."""
        query = select(Event).where(
            Event.lon.between(bbox.min_lon, bbox.max_lon),
            Event.lat.between(bbox.min_lat, bbox.max_lat),
        )
        if event_type is not None:
            query = query.where(Event.event_type == event_type)
        if until is not None:
            query = query.where(Event.ts <= _as_utc(until))
        query = query.order_by(Event.ts.desc(), Event.id).limit(limit)
        return [_event_view(e) for e in self._session.execute(query).scalars()]

    def get(self, event_id: int) -> EventView | None:
        """One event by id."""
        event = self._session.get(Event, event_id)
        return None if event is None else _event_view(event)

    def for_scene(self, scene_id: int, run: str) -> list[EventView]:
        """One fusion run's events of a pass, in the order they were stored."""
        query = select(Event).where(Event.scene_id == scene_id, Event.run == run).order_by(Event.id)
        return [_event_view(e) for e in self._session.execute(query).scalars()]

    def scene_ids(self, run: str) -> list[int]:
        """The passes holding events of one fusion run."""
        query = (
            select(Event.scene_id)
            .where(Event.run == run, Event.scene_id.is_not(None))
            .distinct()
            .order_by(Event.scene_id)
        )
        return list(self._session.execute(query).scalars())

    def update_evidence(self, event_id: int, *, expected: dict, evidence: dict) -> bool:
        """Replace an event's card, e.g. to record its radar chip, only if it is still ``expected``.

        Compare and set: re-fusing a pass deletes its events and SQLite may give
        their ids to new ones, so a writer holding an old card changes nothing
        (False). A card is never emptied.
        """
        if not evidence:
            raise ValueError(f"event {event_id} cannot lose its evidence")
        event = self._session.get(Event, event_id)
        if event is None or event.evidence != expected:
            return False
        event.evidence = dict(evidence)
        self._session.flush()
        return True


def _event_view(e: Event) -> EventView:
    return EventView(
        id=e.id,
        event_type=e.event_type,
        ts=_as_utc(e.ts),
        lon=e.lon,
        lat=e.lat,
        severity=e.severity,
        detection_id=e.detection_id,
        scene_id=e.scene_id,
        run=e.run,
        evidence=e.evidence,
        acknowledged=e.acknowledged,
    )


@dataclass(frozen=True)
class AnalysedWindowView:
    """An analysed window as the API serves it, with its pass."""

    id: int
    scene_id: int
    product_name: str
    satellite: str
    acq_time: dt.datetime
    detector: str
    detector_version: str | None
    polarisation: str | None
    # Detections stored now inside the window for its pass, detector and polarisation.
    detections: int
    bbox: BBox
    # The pass footprint, which the searched area is clipped to.
    footprint_wkt: str | None


class AnalysedWindowRepository:
    """Where a detector searched: one record per pass, detector, polarisation and box (PLAN §10)."""

    # A re-run recomputes its box from the same centre and size: equal far below a metre.
    SAME_BOX_DEG = 1e-6

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        scene_id: int,
        bbox: BBox,
        *,
        detector: str,
        detector_version: str | None,
        polarisation: str,
        detections: int,
    ) -> None:
        """Record a run over a window, replacing an earlier run of the detector and channel over that box.

        Keyed like ``DetectionRepository.replace_in_bbox`` (pass, detector,
        polarisation), plus the box itself.
        """
        if detections < 0:
            raise ValueError(f"a window cannot hold {detections} detections")
        same_box = [
            func.abs(column - value) < self.SAME_BOX_DEG
            for column, value in (
                (AnalysedWindow.min_lon, bbox.min_lon),
                (AnalysedWindow.min_lat, bbox.min_lat),
                (AnalysedWindow.max_lon, bbox.max_lon),
                (AnalysedWindow.max_lat, bbox.max_lat),
            )
        ]
        self._session.query(AnalysedWindow).filter(
            AnalysedWindow.scene_id == scene_id,
            AnalysedWindow.detector == detector,
            AnalysedWindow.polarisation == polarisation,
            *same_box,
        ).delete(synchronize_session=False)
        self._session.add(
            AnalysedWindow(
                scene_id=scene_id,
                detector=detector,
                detector_version=detector_version,
                polarisation=polarisation,
                min_lon=bbox.min_lon,
                min_lat=bbox.min_lat,
                max_lon=bbox.max_lon,
                max_lat=bbox.max_lat,
                detections=detections,
            )
        )
        self._session.flush()

    def recent(
        self,
        bbox: BBox,
        *,
        since: dt.datetime,
        until: dt.datetime | None = None,
        limit: int = 2_000,
    ) -> list[AnalysedWindowView]:
        """Windows overlapping ``bbox`` of passes acquired since ``since``, newest pass first.

        Passes with events are included whatever their age: their dark-vessel
        candidates are always shown, and so must be the area they came from.
        Counts are of the detections stored now, which a later overlapping run
        may have replaced.
        """
        stored_now = (
            select(func.count(Detection.id))
            .where(
                Detection.scene_id == AnalysedWindow.scene_id,
                Detection.detector == AnalysedWindow.detector,
                Detection.polarisation == AnalysedWindow.polarisation,
                Detection.lon.between(AnalysedWindow.min_lon, AnalysedWindow.max_lon),
                Detection.lat.between(AnalysedWindow.min_lat, AnalysedWindow.max_lat),
            )
            .correlate(AnalysedWindow)
            .scalar_subquery()
        )
        fused = select(Event.scene_id).where(Event.scene_id.is_not(None))
        query = (
            select(AnalysedWindow, SarScene, stored_now)
            .join(SarScene, SarScene.id == AnalysedWindow.scene_id)
            .where(
                AnalysedWindow.max_lon >= bbox.min_lon,
                AnalysedWindow.min_lon <= bbox.max_lon,
                AnalysedWindow.max_lat >= bbox.min_lat,
                AnalysedWindow.min_lat <= bbox.max_lat,
                or_(SarScene.acq_time >= _as_utc(since), AnalysedWindow.scene_id.in_(fused)),
                *([SarScene.acq_time <= _as_utc(until)] if until is not None else []),
            )
            .order_by(SarScene.acq_time.desc(), AnalysedWindow.id)
            .limit(limit)
        )
        return [
            AnalysedWindowView(
                id=window.id,
                scene_id=window.scene_id,
                product_name=scene.product_name,
                satellite=scene.satellite,
                acq_time=_as_utc(scene.acq_time),
                detector=window.detector,
                detector_version=window.detector_version,
                polarisation=window.polarisation,
                detections=count,
                bbox=BBox(window.min_lon, window.min_lat, window.max_lon, window.max_lat),
                footprint_wkt=scene.footprint_wkt,
            )
            for window, scene, count in self._session.execute(query).all()
        ]
