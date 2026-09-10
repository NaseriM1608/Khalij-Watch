"""Phase 0 access checks.

Verifies that every external data source this project depends on is reachable
and functional from THIS machine, before any code is built on top of it.

Run:  uv run --python 3.12 python scripts/check_access.py
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

TIMEOUT = 60
# Some hosts (aisstream.io) sit behind Cloudflare and 403 the default
# Python user-agent. Present a browser-like UA for plain reachability probes.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
RETRIES = 3
RETRY_BACKOFF_S = 2
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)

# Persian Gulf sample points (lon, lat)
GULF_POINTS = {
    "Hormuz": (56.5, 26.5),
    "Central Gulf": (52.0, 27.0),
    "N Gulf (Kuwait)": (48.5, 29.5),
}

# Minimum acquisitions in 30 days before we consider coverage adequate.
MIN_ACQUISITIONS_30D = 5


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [f"[{mark}] {self.name}: {self.detail}"]
        lines.extend(f"       {n}" for n in self.notes)
        return "\n".join(lines)


def _open(req: urllib.request.Request, retries: int = RETRIES):
    """Open a request, retrying transient DNS/network failures.

    Local DNS here has proven intermittent, so a single failure must not be
    reported as "this data source is unavailable".
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=TIMEOUT)
        except urllib.error.HTTPError:
            raise  # a real HTTP response; the caller decides what it means
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    assert last is not None
    raise last


def _get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    with _open(req) as resp:
        return json.load(resp)


def _http_status(url: str, method: str = "GET") -> int:
    """Return the HTTP status, or 0 if the host could not be reached at all."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with _open(req) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def query_sentinel1(lon: float, lat: float, days: int = 30) -> list[dict]:
    """Search CDSE for Sentinel-1 IW GRDH products over a point. No auth needed."""
    since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    odata_filter = (
        "Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') and "
        f"ContentDate/Start gt {since} and contains(Name,'IW_GRDH')"
    )
    query = urllib.parse.urlencode(
        {
            "$filter": odata_filter,
            "$top": "100",
            "$orderby": "ContentDate/Start desc",
        }
    )
    return _get_json(f"{CDSE_ODATA}?{query}").get("value", [])


def mean_revisit_days(products: list[dict]) -> float:
    """Mean gap between distinct acquisitions, in days."""
    stamps = sorted(
        dt.datetime.strptime(p["ContentDate"]["Start"][:19], "%Y-%m-%dT%H:%M:%S") for p in products
    )
    # Products from one pass share a timestamp within minutes; collapse them.
    gaps = [
        (b - a).total_seconds() / 86400
        for a, b in itertools.pairwise(stamps)
        if (b - a).total_seconds() > 3600
    ]
    return sum(gaps) / len(gaps) if gaps else 0.0


def check_cdse_catalogue() -> Result:
    notes: list[str] = []
    total = 0
    try:
        for name, (lon, lat) in GULF_POINTS.items():
            products = query_sentinel1(lon, lat)
            total += len(products)
            revisit = mean_revisit_days(products)
            sats = sorted({p["Name"][:3] for p in products})
            pols = sorted({p["Name"][14:18] for p in products})
            notes.append(
                f"{name:16s} n={len(products):3d}  "
                f"revisit={revisit:.1f}d  sats={','.join(sats) or '-'}  "
                f"pol={','.join(pols) or '-'}"
            )
    except Exception as exc:
        return Result("CDSE catalogue (Sentinel-1)", False, f"{type(exc).__name__}: {exc}")

    ok = total >= MIN_ACQUISITIONS_30D
    detail = f"{total} IW_GRDH products over 3 Gulf points in 30 days"
    return Result("CDSE catalogue (Sentinel-1)", ok, detail, notes)


def check_cdse_credentials() -> Result:
    """Confirm a token can be obtained, if credentials are configured."""
    user = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not user or not password:
        return Result(
            "CDSE credentials",
            False,
            "CDSE_USERNAME / CDSE_PASSWORD not set - register at "
            "https://dataspace.copernicus.eu (free) and add them to .env",
        )
    body = urllib.parse.urlencode(
        {
            "client_id": "cdse-public",
            "username": user,
            "password": password,
            "grant_type": "password",
        }
    ).encode()
    try:
        req = urllib.request.Request(CDSE_TOKEN, data=body, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            token = json.load(resp).get("access_token", "")
        return Result("CDSE credentials", bool(token), f"token acquired ({len(token)} chars)")
    except Exception as exc:
        return Result("CDSE credentials", False, f"{type(exc).__name__}: {exc}")


def check_aisstream() -> Result:
    key = os.environ.get("AISSTREAM_API_KEY")
    status = _http_status("https://aisstream.io/")
    if status != 200:
        return Result(
            "aisstream.io",
            False,
            f"site unreachable ({'DNS/network failure' if status == 0 else f'HTTP {status}'})",
        )
    if not key:
        return Result(
            "aisstream.io",
            False,
            "reachable, but AISSTREAM_API_KEY not set - get a free key at "
            "https://aisstream.io/authenticate and add it to .env",
        )
    return Result(
        "aisstream.io",
        True,
        "reachable and key present (live socket tested in Phase 1)",
    )


def check_xview3() -> Result:
    """xView3 download is US export-control gated; reachability != eligibility."""
    status = _http_status("https://iuu.xview.us/")
    if status != 200:
        return Result(
            "xView3-SAR",
            False,
            f"iuu.xview.us unreachable ({status or 'DNS/network'}) - use the LS-SSDD fallback",
        )
    return Result(
        "xView3-SAR",
        True,
        "iuu.xview.us reachable (HTTP 200)",
        [
            "Reachability is NOT eligibility. Download is US export-control gated.",
            "ACTION: register manually and confirm the download works.",
            "Fallback if refused: LS-SSDD-v1.0 + Official-SSDD (both Apache-2.0).",
        ],
    )


def main() -> int:
    print("=" * 74)
    print("PHASE 0 ACCESS CHECKS - khalij-watch")
    print(f"UTC: {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    results = [
        check_cdse_catalogue(),
        check_cdse_credentials(),
        check_aisstream(),
        check_xview3(),
    ]
    for result in results:
        print(result.render())

    blocking = [r for r in results if not r.ok]
    print("-" * 74)
    if blocking:
        print(f"{len(blocking)} of {len(results)} checks need attention:")
        for r in blocking:
            print(f"  - {r.name}")
        print("\nMost are one-time manual signups. See .env.example.")
    else:
        print("All access checks passed.")
    # Catalogue access is the only hard blocker; the rest are user signups.
    return 0 if results[0].ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
