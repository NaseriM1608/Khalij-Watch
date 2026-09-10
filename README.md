# khalij-watch

Persian Gulf maritime domain awareness. Ingests live AIS and free Sentinel-1 SAR
imagery, uses deep learning to detect and characterise vessels in the radar imagery,
and fuses the two to surface **dark vessels** — ships the radar sees that broadcast
no matching AIS.

> **Design premise:** SAR is the truth layer. AIS is a claim that may be lying.

## Why that framing

Free AIS over the Gulf comes from roughly four contributing shore stations (3 UAE,
1 Qatar, none on the Iranian side) against a 200–300 km-wide basin, and 2026 saw
severe GNSS jamming around Hormuz. The AIS layer is **port and anchorage awareness,
not Gulf-wide coverage**, and much of what arrives is fabricated. The product is the
disagreement between what ships claim and what radar observes.

## What the deep learning does — and does not do

Pixels give **existence, size, and heading**. AIS and registries give **identity and
type**. Fine-grained vessel type is *not* inferable from 10 m Sentinel-1 pixels: at
that resolution a 300 m tanker is 30×5 px. The model outputs `is_vessel`, length,
width, heading and a coarse class; anything finer comes from the AIS/registry join.

Detection uses a **dense stride-2 point detector**, not oriented bounding boxes —
there is no meaningful orientation to regress on a 3 px target.

## Status

Phase 0 (foundation and access checks) — in progress. See `PLAN.md` for the full
plan and `PROMPT.md` for the build specification.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned.

```bash
cp .env.example .env      # then fill in credentials
uv sync
uv run python scripts/check_access.py
```

## Access checks

`scripts/check_access.py` verifies every external dependency is reachable from this
machine before anything is built on top of it. Run it first, and whenever something
downstream behaves strangely.

Measured 2026-09-10 (30-day window, `IW_GRDH_1S`, all dual-pol VV+VH, S1C+S1D):

| Point | Acquisitions | Mean revisit |
|---|---|---|
| Strait of Hormuz | 28 | **2.0 d** |
| Central Gulf | 20 | 2.8 d |
| N Gulf (Kuwait) | 20 | 3.2 d |

## Tests

```bash
uv run pytest -m "not network"
```

Network-dependent tests are marked `network` and validate the live CDSE data path.

## Attribution

Contains modified Copernicus Sentinel data. Copernicus data is free and open with
attribution. Other sources carry stricter terms — AIS feed `source` values must be
propagated to the UI, and Global Fishing Watch data is CC BY-NC 4.0 (non-commercial).
