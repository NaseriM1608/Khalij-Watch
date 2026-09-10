"""Seed synthetic Gulf traffic so the map can be developed without credentials.

Every row is written with ``source="synthetic"``. That label is load-bearing:
it flows through the repository to the API and into the UI, so simulated
traffic can never be mistaken for a real observation.

Run:  uv run python scripts/seed_demo.py --hours 6 --vessels 120
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ais.messages import AisPositionReport, VesselStatic
from core.db.repository import PositionRepository, VesselRepository
from core.db.session import (
    create_all,
    create_db_engine,
    make_session_factory,
    session_scope,
)

SOURCE = "synthetic"

# The main deep-water route, Strait of Hormuz to the northern Gulf.
SHIPPING_LANE = [
    (56.8, 26.4),
    (56.2, 26.6),
    (55.2, 26.2),
    (54.0, 25.8),
    (52.5, 26.3),
    (51.0, 27.2),
    (49.8, 28.3),
    (48.6, 29.4),
]

# Anchorages, where vessels sit near-stationary in dense clusters.
ANCHORAGES = {
    "Fujairah": (56.40, 25.18),
    "Bandar Abbas": (56.20, 27.12),
    "Jebel Ali": (55.03, 24.98),
    "Ras Tanura": (50.16, 26.70),
    "Kharg Island": (50.32, 29.23),
}

# MID prefixes for Gulf-littoral flags, so MMSIs look regionally plausible.
GULF_MIDS = [403, 408, 422, 425, 447, 461, 466, 470]

SHIP_TYPES = [
    (80, "cargo_tanker", 0.34),  # tanker
    (70, "cargo_tanker", 0.26),  # cargo
    (30, "fishing", 0.16),
    (52, "service_tug", 0.12),
    (60, "passenger_other", 0.12),
]

NAME_PREFIXES = ["AL", "GULF", "PACIFIC", "DESERT", "PEARL", "STAR", "OCEAN", "SEA"]
NAME_SUFFIXES = ["VOYAGER", "PIONEER", "TRADER", "SPIRIT", "HORIZON", "GLORY", "DAWN"]


@dataclass
class SimVessel:
    mmsi: int
    name: str
    ship_type: int
    length_m: float
    beam_m: float
    speed_kn: float
    anchored: bool
    progress: float  # position along the lane, 0..1
    direction: int  # +1 outbound, -1 inbound
    anchor: tuple[float, float] | None


def _interpolate_lane(progress: float) -> tuple[float, float, float]:
    """Position and course along the lane at ``progress`` in [0, 1]."""
    clamped = min(max(progress, 0.0), 1.0)
    span = clamped * (len(SHIPPING_LANE) - 1)
    index = min(int(span), len(SHIPPING_LANE) - 2)
    frac = span - index
    lon0, lat0 = SHIPPING_LANE[index]
    lon1, lat1 = SHIPPING_LANE[index + 1]
    lon = lon0 + (lon1 - lon0) * frac
    lat = lat0 + (lat1 - lat0) * frac
    # Course over ground, accounting for longitude convergence.
    course = math.degrees(
        math.atan2(
            (lon1 - lon0) * math.cos(math.radians(lat)),
            lat1 - lat0,
        )
    )
    return lon, lat, course % 360.0


def _make_vessel(rng: random.Random, index: int) -> SimVessel:
    ship_type = rng.choices([t[0] for t in SHIP_TYPES], weights=[t[2] for t in SHIP_TYPES])[0]
    anchored = rng.random() < 0.28
    if ship_type == 30:  # fishing vessels are small and slow
        length = rng.uniform(12, 40)
        speed = rng.uniform(2, 8)
    elif ship_type in (70, 80):
        length = rng.uniform(150, 330)
        speed = rng.uniform(10, 16)
    else:
        length = rng.uniform(30, 90)
        speed = rng.uniform(6, 12)

    return SimVessel(
        mmsi=rng.choice(GULF_MIDS) * 1_000_000 + rng.randint(100_000, 999_999),
        name=f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES)} {index % 90 + 10}",
        ship_type=ship_type,
        length_m=round(length, 1),
        beam_m=round(length / rng.uniform(5.5, 7.5), 1),
        speed_kn=round(0.0 if anchored else speed, 1),
        anchored=anchored,
        progress=rng.random(),
        direction=rng.choice([1, -1]),
        anchor=(rng.choice(list(ANCHORAGES.values())) if anchored else None),
    )


def _position_at(
    vessel: SimVessel, step: int, total_steps: int, rng: random.Random
) -> tuple[float, float, float]:
    if vessel.anchored and vessel.anchor is not None:
        # Swing gently around the anchor rather than sitting perfectly still.
        lon, lat = vessel.anchor
        jitter = 0.012
        angle = (step / max(total_steps, 1)) * math.tau + vessel.progress * math.tau
        return (
            lon + math.cos(angle) * jitter + rng.uniform(-0.002, 0.002),
            lat + math.sin(angle) * jitter + rng.uniform(-0.002, 0.002),
            (math.degrees(angle) + 90) % 360,
        )

    # Roughly: 1 degree of latitude is 60 nautical miles.
    lane_length_deg = 11.0
    travelled = (vessel.speed_kn * step / 60.0) / (lane_length_deg * 60.0)
    progress = (vessel.progress + vessel.direction * travelled) % 1.0
    lon, lat, course = _interpolate_lane(progress)
    if vessel.direction < 0:
        course = (course + 180) % 360
    return lon + rng.uniform(-0.01, 0.01), lat + rng.uniform(-0.01, 0.01), course


def seed(
    *, vessel_count: int, hours: int, interval_minutes: int, seed_value: int
) -> tuple[int, int]:
    rng = random.Random(seed_value)
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    steps = max(1, (hours * 60) // interval_minutes)

    fleet = [_make_vessel(rng, i) for i in range(vessel_count)]
    # MMSIs must be unique or the fleet silently collapses on upsert.
    fleet = list({v.mmsi: v for v in fleet}.values())

    engine = create_db_engine()
    create_all(engine)
    factory = make_session_factory(engine)

    reports: list[AisPositionReport] = []
    for vessel in fleet:
        for step in range(steps, -1, -1):
            ts = now - dt.timedelta(minutes=step * interval_minutes)
            lon, lat, course = _position_at(vessel, steps - step, steps, rng)
            reports.append(
                AisPositionReport(
                    mmsi=vessel.mmsi,
                    ts=ts,
                    lon=round(lon, 6),
                    lat=round(lat, 6),
                    source=SOURCE,
                    sog_kn=vessel.speed_kn,
                    cog_deg=round(course, 1),
                    heading_deg=round(course, 1),
                    nav_status=1 if vessel.anchored else 0,
                    msg_type=1,
                )
            )

    with session_scope(factory) as session:
        vessels_repo = VesselRepository(session)
        vessels_repo.ensure_exists({v.mmsi for v in fleet}, seen_at=now)
        session.flush()
        for vessel in fleet:
            vessels_repo.upsert_static(
                VesselStatic(
                    mmsi=vessel.mmsi,
                    name=vessel.name,
                    ship_type=vessel.ship_type,
                    length_m=vessel.length_m,
                    beam_m=vessel.beam_m,
                ),
                seen_at=now,
            )
        session.flush()
        written = PositionRepository(session).add_batch(reports)

    return len(fleet), written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vessels", type=int, default=120)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fleet_size, written = seed(
        vessel_count=args.vessels,
        hours=args.hours,
        interval_minutes=args.interval_minutes,
        seed_value=args.seed,
    )
    print(f"seeded {fleet_size} vessels, {written} positions over {args.hours}h")
    print(f'all rows tagged source="{SOURCE}" - they are NOT real observations')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
