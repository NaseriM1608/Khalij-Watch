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
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load .env before any check reads the environment.

    Without this the script reports credentials as missing when they are
    sitting in the file, which is a needlessly confusing failure.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


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
    """Confirm a token can actually be obtained.

    Delegates to TokenManager so this checks the SAME code path the ingest
    uses, rather than a second implementation that could drift from it.
    """
    from core.sar.auth import AuthError, TokenManager

    manager = TokenManager()
    if not manager.is_configured:
        return Result(
            "CDSE credentials",
            False,
            "not set - create an OAuth client at https://dataspace.copernicus.eu "
            "and put its id and secret in CDSE_CLIENT_ID / CDSE_CLIENT_SECRET",
        )
    try:
        token = manager.token()
    except AuthError as exc:
        return Result(
            "CDSE credentials",
            False,
            str(exc),
            [f"attempted the {manager.grant_type} grant"],
        )
    return Result(
        "CDSE credentials",
        bool(token),
        f"token acquired via {manager.grant_type} grant ({len(token)} chars)",
    )


# How long the pixel probe may run before it is abandoned. Covers a catalogue
# query, an object listing and one windowed read over a slow link.
PIXEL_PROBE_TIMEOUT_S = 120
# How long the probe gets to exit by itself once it has reported.
PIXEL_PROBE_EXIT_GRACE_S = 3


def result_from_verdict(
    verdict: dict | None, *, timed_out: bool = False, stderr_tail: str = ""
) -> Result:
    """Translate the pixel probe's JSON verdict into a check result."""
    from core.sar.reader import REMEDIES

    name = "CDSE pixel access"
    if timed_out:
        return Result(
            name,
            False,
            f"no answer within {PIXEL_PROBE_TIMEOUT_S}s",
            ["object storage or the catalogue is unreachable or very slow; retry"],
        )
    if verdict is None:
        tail = stderr_tail.strip().splitlines()[-4:] if stderr_tail else []
        return Result(name, False, "the probe process ended without reporting a result", tail)

    if verdict.get("ok"):
        shape = "x".join(str(v) for v in verdict.get("shape", []))
        return Result(
            name,
            True,
            f"read a {shape} window of real pixels in {verdict.get('seconds')}s "
            "through the production reader",
            [f"scene: {verdict.get('product')}", f"max DN in window: {verdict.get('max_dn')}"],
        )

    reason = str(verdict.get("reason") or "other")
    notes = [REMEDIES.get(reason, REMEDIES["other"])]
    detail = str(verdict.get("detail") or "")
    if detail:
        notes.append(f"detail: {detail[:200]}")
    return Result(name, False, f"S3 keys are set but reading pixels failed ({reason})", notes)


def check_s3_pixel_read(timeout_s: float = PIXEL_PROBE_TIMEOUT_S) -> Result:
    """Read real pixels from a recent Gulf scene through the production reader.

    Runs in a child process, for two measured reasons. After a failed request
    GDAL on this host can leave the process unable to exit - even ``os._exit``
    does not return - so the child is terminated from outside once it has
    reported; killing it skips the native teardown that hangs. And a stalled
    network must not freeze the whole check.
    """
    import json
    import subprocess
    import threading

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        "from core.sar.reader import probe_main; probe_main()"
    )
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return Result("CDSE pixel access", False, f"could not start the probe: {exc}")

    reported: list[str] = []
    stderr_lines: list[str] = []
    stdout_reader = threading.Thread(
        target=lambda: reported.append(child.stdout.readline()), daemon=True
    )
    stderr_reader = threading.Thread(target=lambda: stderr_lines.extend(child.stderr), daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    stdout_reader.join(timeout_s)
    timed_out = stdout_reader.is_alive()

    try:
        child.wait(timeout=PIXEL_PROBE_EXIT_GRACE_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    stderr_reader.join(1.0)

    verdict = None
    if reported and reported[0].strip():
        try:
            verdict = json.loads(reported[0])
        except json.JSONDecodeError:
            verdict = None
    return result_from_verdict(verdict, timed_out=timed_out, stderr_tail="".join(stderr_lines))


def check_cdse_download() -> Result:
    """Confirm the credentials can actually READ PIXELS, not merely authenticate.

    These are separate entitlements at CDSE, and conflating them hides a real
    failure: a Sentinel Hub OAuth client authenticates happily and is then
    refused by the download service with "Token audience not allowed". A check
    that stops at token acquisition reports PASS while nothing can be read.
    """
    import urllib.error

    from core.sar.auth import AuthError, S3Credentials, TokenManager

    if S3Credentials.from_env() is not None:
        # Presence proves nothing: an earlier version returned PASS here for
        # keys that had never been used, while the read path itself crashed.
        return check_s3_pixel_read()

    manager = TokenManager()
    if not manager.is_configured:
        return Result("CDSE pixel access", False, "no credentials configured")

    try:
        token = manager.token()
    except AuthError as exc:
        return Result("CDSE pixel access", False, str(exc))

    # Probe a REAL product with a 100-byte range: the smallest request that
    # genuinely exercises the download entitlement. Uses this script's own
    # stdlib catalogue query, so the checker stays dependency-free and can be
    # run before the project is installed.
    lon, lat = GULF_POINTS["Hormuz"]
    try:
        products = query_sentinel1(lon, lat, days=20)
    except Exception as exc:
        return Result("CDSE pixel access", False, f"catalogue lookup failed: {exc}")
    if not products:
        return Result("CDSE pixel access", False, "no product available to probe")

    probe = urllib.request.Request(
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({products[0]['Id']})/$value",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Range": "bytes=0-99",
        },
    )
    try:
        _open(probe)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Result(
                "CDSE pixel access",
                False,
                f"authenticated, but the download service refused the token (HTTP {exc.code})",
                [
                    "A Sentinel Hub OAuth client (id starting 'sh-') can query the",
                    "catalogue but NOT download product data.",
                    "FIX: in the CDSE dashboard generate S3 credentials, then set",
                    "     CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY in .env.",
                    "That is also the only path that supports windowed byte-range",
                    "reads, which this project depends on.",
                ],
            )
        return Result("CDSE pixel access", False, f"HTTP {exc.code}")
    except Exception as exc:
        return Result("CDSE pixel access", False, f"{type(exc).__name__}: {exc}")
    return Result("CDSE pixel access", True, "download service accepted the token")


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
    _load_dotenv()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print("=" * 74)
    print("PHASE 0 ACCESS CHECKS - khalij-watch")
    print(f"UTC: {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    results = [
        check_cdse_catalogue(),
        check_cdse_credentials(),
        check_cdse_download(),
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
