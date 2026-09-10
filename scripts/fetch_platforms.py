"""Fetch the Offshore Platform Dataset and trim it to the AOI.

The Gulf holds thousands of fixed oil and gas structures. Each is a bright
permanent point target that a CFAR cannot distinguish from a stationary
vessel, so this mask is what stops them dominating the detections.

Source: OPD v1.0.0, Spanier et al. (2026), Big Earth Data.
        https://doi.org/10.5281/zenodo.19046824    CC BY 4.0

Run:  uv run python scripts/fetch_platforms.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db.spatial import GULF_BBOX
from core.net import USER_AGENT, open_with_retry
from core.sar.infrastructure import (
    ATTRIBUTION,
    DEFAULT_PATH,
    PERSIAN_GULF,
    load_platforms,
)

ZENODO_RECORD = "19046824"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
# The unique-platform file: one entry per structure, with its lifespan. Far
# smaller than the per-quarter file and exactly what a time-aware mask needs.
WANTED_FILE = "OPD_platforms.geojson"


def _fetch(url: str) -> bytes:
    # Zenodo refuses the default Python user-agent.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with open_with_retry(request, timeout=300) as response:
        return response.read()


def download(destination: Path) -> Path:
    """Download the platform GeoJSON, resolving its URL from the Zenodo API."""
    record = json.loads(_fetch(ZENODO_API))
    for entry in record.get("files", []):
        if entry.get("key") == WANTED_FILE:
            url = entry["links"]["self"]
            break
    else:
        raise RuntimeError(f"{WANTED_FILE} is not in Zenodo record {ZENODO_RECORD}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_fetch(url))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download if present")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    if args.path.is_file() and not args.force:
        print(f"{args.path} already present; use --force to re-download")
    else:
        print(f"downloading {WANTED_FILE} from Zenodo record {ZENODO_RECORD}...")
        download(args.path)
        print(f"saved {args.path} ({args.path.stat().st_size / 1e6:.2f} MB)")

    gulf = load_platforms(args.path, region=PERSIAN_GULF, bbox=GULF_BBOX)
    still_present = sum(1 for p in gulf if p.last_observed is None)
    deepest = min((p.water_depth_m or 0.0) for p in gulf) if gulf else 0.0

    print()
    print(f"Persian Gulf platforms in AOI : {len(gulf)}")
    print(f"  still present at end of record: {still_present}")
    print(f"  deepest water                 : {abs(deepest):.0f} m")
    by_eez: dict[str, int] = {}
    for platform in gulf:
        key = platform.country_eez or "unknown"
        by_eez[key] = by_eez.get(key, 0) + 1
    for eez, count in sorted(by_eez.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {eez:24s} {count}")
    print()
    print(ATTRIBUTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
