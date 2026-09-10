"""Database schema for khalij-watch.

Portability note
----------------
The project currently runs on SQLite and is expected to move to
PostgreSQL + PostGIS later. To keep that migration contained:

* Geometry is stored as plain ``lon`` / ``lat`` REAL columns rather than a
  vendor geometry type. PostGIS gains a real ``geometry(Point, 4326)`` column
  in a later migration, backfilled from these.
* Spatial lookups go through an R*Tree companion table on SQLite (see
  ``core.db.spatial``); on PostGIS they become GiST indexes.
* No query is written against a table directly outside ``core.db.repository``.

Longitude/latitude are always WGS84 (EPSG:4326), in degrees.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SRID_WGS84 = 4326


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Vessel(Base):
    """A vessel identity, keyed by MMSI and enriched from registries."""

    __tablename__ = "vessels"

    mmsi: Mapped[int] = mapped_column(Integer, primary_key=True)
    imo: Mapped[int | None] = mapped_column(Integer, index=True)
    name: Mapped[str | None] = mapped_column(String(128))
    callsign: Mapped[str | None] = mapped_column(String(16))
    ship_type: Mapped[int | None] = mapped_column(Integer)
    coarse_class: Mapped[str | None] = mapped_column(String(24), index=True)
    length_m: Mapped[float | None] = mapped_column(Float)
    beam_m: Mapped[float | None] = mapped_column(Float)
    flag: Mapped[str | None] = mapped_column(String(8), index=True)
    first_seen: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    sanctioned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    sanction_source: Mapped[str | None] = mapped_column(String(64))

    positions: Mapped[list[AisPosition]] = relationship(back_populates="vessel")


class AisPosition(Base):
    """One AIS position report.

    ``plausibility_score`` and ``anomaly_flags`` are populated by the ingest
    filter. Implausible positions are FLAGGED AND KEPT, never dropped: in this
    AOI a fabricated track is itself evidence.
    """

    __tablename__ = "ais_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mmsi: Mapped[int] = mapped_column(ForeignKey("vessels.mmsi"), nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    sog_kn: Mapped[float | None] = mapped_column(Float)
    cog_deg: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    nav_status: Mapped[int | None] = mapped_column(Integer)
    msg_type: Mapped[int | None] = mapped_column(Integer)
    # Required for attribution: aiscast re-serves per-source licences.
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    plausibility_score: Mapped[float | None] = mapped_column(Float, index=True)
    # none_as_null: without it a clean row stores the JSON literal `null`,
    # which still satisfies "IS NOT NULL" and silently breaks anomaly queries.
    anomaly_flags: Mapped[list | None] = mapped_column(JSON(none_as_null=True))

    vessel: Mapped[Vessel] = relationship(back_populates="positions")

    __table_args__ = (
        Index("ix_ais_positions_mmsi_ts", "mmsi", "ts"),
        Index("ix_ais_positions_ts", "ts"),
        UniqueConstraint("mmsi", "ts", "lon", "lat", name="uq_ais_position"),
    )


class SarScene(Base):
    """A Sentinel-1 product covering part of the AOI.

    ``satellite`` is first-class because CFAR and detector parameters do NOT
    transfer between units. Global Fishing Watch's production detector needed
    full recalibration for S1C/S1D and was offline for two months as a result.
    """

    __tablename__ = "sar_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    product_name: Mapped[str] = mapped_column(String(256), nullable=False)
    satellite: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(8), default="IW")
    polarisation: Mapped[str | None] = mapped_column(String(8))
    acq_time: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    orbit_direction: Mapped[str | None] = mapped_column(String(16))
    # Footprint bounds, for AOI intersection without a geometry type.
    min_lon: Mapped[float] = mapped_column(Float, nullable=False)
    min_lat: Mapped[float] = mapped_column(Float, nullable=False)
    max_lon: Mapped[float] = mapped_column(Float, nullable=False)
    max_lat: Mapped[float] = mapped_column(Float, nullable=False)
    footprint_wkt: Mapped[str | None] = mapped_column(String(4096))
    cog_url: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(24), default="discovered", index=True)
    processed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    detections: Mapped[list[Detection]] = relationship(back_populates="scene")


class Detection(Base):
    """A vessel-like object found in a SAR scene.

    Note what is absent: there is no fine-grained vessel type. At 10 m pixel
    spacing that signal is not in the data. ``pred_class`` is a coarse,
    honestly-reported secondary output.
    """

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("sar_scenes.id"), nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    length_m: Mapped[float | None] = mapped_column(Float)
    width_m: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_vessel_prob: Mapped[float | None] = mapped_column(Float)
    is_fishing_prob: Mapped[float | None] = mapped_column(Float)
    pred_class: Mapped[str | None] = mapped_column(String(24))
    class_probs: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    chip_path: Mapped[str | None] = mapped_column(String(512))
    detector: Mapped[str] = mapped_column(String(64), nullable=False, default="cfar")
    detector_version: Mapped[str | None] = mapped_column(String(32))

    scene: Mapped[SarScene] = relationship(back_populates="detections")
    match: Mapped[FusionMatch | None] = relationship(back_populates="detection", uselist=False)

    __table_args__ = (Index("ix_detections_scene", "scene_id"),)


class FusionMatch(Base):
    """Link between a SAR detection and an AIS track, with its evidence.

    A detection with no row here, once infrastructure is excluded, is a
    DARK VESSEL: the headline output of the system.
    """

    __tablename__ = "fusion_matches"

    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), primary_key=True)
    mmsi: Mapped[int] = mapped_column(ForeignKey("vessels.mmsi"), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    distance_m: Mapped[float | None] = mapped_column(Float)
    # Azimuth displacement applied before matching. Sentinel-1 IW shifts a
    # moving target by roughly 100 m along-track per 1 m/s of radial velocity,
    # so a 12 kn vessel can sit 500-700 m from its reported AIS position.
    azimuth_shift_m: Mapped[float | None] = mapped_column(Float)
    length_error_m: Mapped[float | None] = mapped_column(Float)
    heading_error_deg: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(32), default="hungarian")
    matched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    detection: Mapped[Detection] = relationship(back_populates="match")


class Event(Base):
    """An analyst-facing finding. ``evidence`` must always be populated."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    lon: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16), default="info", index=True)
    detection_id: Mapped[int | None] = mapped_column(ForeignKey("detections.id"))
    mmsi: Mapped[int | None] = mapped_column(Integer, index=True)
    # Never surface an alert whose evidence cannot be shown to the analyst.
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class Zone(Base):
    """A user-drawn geofence."""

    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    polygon_wkt: Mapped[str] = mapped_column(String(8192), nullable=False)
    rule: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
