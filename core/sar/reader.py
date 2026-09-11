"""Open Sentinel-1 measurement rasters straight from CDSE object storage.

This is the production read path, and the pixel-access check exercises it
deliberately instead of testing credentials some other way. An earlier
credential helper passed every unit test while crashing on first contact with
rasterio, because nothing ever opened a raster through it.

Failure handling, from behaviour observed against the live endpoint:

* A wrong access key and a wrong secret produce different S3 error codes,
  ``InvalidAccessKeyId`` and ``SignatureDoesNotMatch``, so the operator is
  told which one to fix.
* DNS on this host fails intermittently, and GDAL reports that as an ordinary
  I/O error ("Could not resolve host"). Such transport failures are retried;
  credential failures are not, because retrying a wrong key only delays the
  message.
* After a failed request, a process that has used GDAL on this host can fail
  to exit at all. It hangs after atexit handlers have run with no Python
  threads alive, and even ``os._exit`` does not return, which points at
  native library teardown. :func:`probe_main` therefore only reports; the
  process that launched it must terminate it rather than wait.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from core.sar.auth import S3_REGION, S3Credentials

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from core.sar.catalogue import SarProduct
    from core.sar.windows import PixelWindow

DEFAULT_ATTEMPTS = 3
BACKOFF_S = 2.0

POLARISATIONS = ("vv", "vh", "hh", "hv")
RASTER_SUFFIXES = (".tif", ".tiff")

# How the pixel-access probe picks and samples a scene.
PROBE_WINDOW_PX = 256
PROBE_LOOKBACK_DAYS = 20

# Substrings GDAL, curl and botocore use for transient transport failures.
TRANSIENT_MARKERS = (
    "Could not resolve host",
    "Couldn't resolve host",
    "Could not connect to the endpoint",
    "Connection timed out",
    "Operation timed out",
    "Read timed out",
    "Failed to connect",
    "Connection reset",
    "Connection aborted",
    "SSL connect error",
    "HTTP response code: 500",
    "HTTP response code: 502",
    "HTTP response code: 503",
    "HTTP response code: 504",
)

REMEDIES = {
    "no_credentials": (
        "set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY in .env; generate them at "
        "https://eodata-s3keysmanager.dataspace.copernicus.eu/"
    ),
    "unknown_access_key": (
        "the access key is not recognised: check it was copied exactly, and that "
        "it has not passed the expiry date chosen when it was created. Regenerate "
        "at https://eodata-s3keysmanager.dataspace.copernicus.eu/"
    ),
    "wrong_secret": (
        "the access key is recognised but the secret does not match it: re-copy "
        "the secret, or generate a new pair (a secret is only shown once)"
    ),
    "access_denied": "the keys were accepted but access to this object was refused",
    "not_found": "the expected object does not exist",
    "network": "object storage could not be reached; DNS on this host is intermittent, so retry",
    "missing_dependency": "reading pixels needs rasterio and boto3; run `uv sync`",
    "other": "unexpected failure; see the detail",
}

# S3 error codes, checked before generic HTTP status text because a
# credential failure also carries a 403.
_S3_ERROR_CODES = {
    "InvalidAccessKeyId": "unknown_access_key",
    "SignatureDoesNotMatch": "wrong_secret",
    "AccessDenied": "access_denied",
    "NoSuchKey": "not_found",
    "NoSuchBucket": "not_found",
}


class S3AccessError(RuntimeError):
    """A request to object storage failed, with a reason an operator can act on."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = f"{reason}: {self.remedy}"
        super().__init__(f"{message} ({detail})" if detail else message)

    @property
    def remedy(self) -> str:
        return REMEDIES.get(self.reason, REMEDIES["other"])

    @property
    def is_transient(self) -> bool:
        return self.reason == "network"


def classify_s3_error(message: str) -> str:
    """Map a GDAL, curl or botocore error message to a reason code."""
    for code, reason in _S3_ERROR_CODES.items():
        if code in message:
            return reason
    if any(marker in message for marker in TRANSIENT_MARKERS):
        return "network"
    if "HTTP response code: 403" in message:
        return "access_denied"
    if "HTTP response code: 404" in message or "No such file" in message:
        return "not_found"
    return "other"


def as_access_error(exc: BaseException) -> S3AccessError:
    """Wrap any failure from GDAL or boto3 as a classified :class:`S3AccessError`."""
    if isinstance(exc, S3AccessError):
        return exc
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
    message = f"{code} {type(exc).__name__}: {exc}".strip()
    return S3AccessError(classify_s3_error(message), message[:300])


def split_s3_path(s3_path: str) -> tuple[str, str]:
    """``/eodata/Sentinel-1/...SAFE`` becomes (``eodata``, ``Sentinel-1/...SAFE``)."""
    bucket, _, prefix = s3_path.strip().lstrip("/").partition("/")
    if not bucket or not prefix:
        raise ValueError(f"not an S3 object path: {s3_path!r}")
    return bucket, prefix.rstrip("/")


def s3_client(creds: S3Credentials) -> Any:
    """A boto3 client for CDSE object storage.

    Path-style addressing is required: virtual-hosted style would prepend the
    bucket name to the endpoint host, which does not resolve.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=creds.endpoint,
        aws_access_key_id=creds.access_key,
        aws_secret_access_key=creds.secret_key,
        region_name=S3_REGION,
        config=Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            retries={"max_attempts": DEFAULT_ATTEMPTS, "mode": "standard"},
        ),
    )


def measurement_keys(
    product: SarProduct, creds: S3Credentials, *, client: Any = None
) -> tuple[str, dict[str, str]]:
    """Locate a product's measurement rasters, keyed by lower-case polarisation.

    Listed rather than derived from the product name: file names carry an
    image number whose assignment depends on the polarisation mode, and a
    guessed key fails far less clearly than a listing does.
    """
    if not product.s3_path:
        raise S3AccessError("not_found", f"{product.name} has no S3 path")
    bucket, prefix = split_s3_path(product.s3_path)
    client = client if client is not None else s3_client(creds)
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/measurement/")
    except Exception as exc:
        raise as_access_error(exc) from exc

    keys: dict[str, str] = {}
    for entry in response.get("Contents", []):
        key = entry["Key"]
        name = key.rsplit("/", 1)[-1].lower()
        if not name.endswith(RASTER_SUFFIXES):
            continue
        for polarisation in POLARISATIONS:
            if f"-{polarisation}-" in name:
                keys[polarisation] = key
    return bucket, keys


def _retrying(
    action: Callable[[], Any],
    *,
    attempts: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> Any:
    """Run ``action``, retrying transient transport failures only."""
    last: S3AccessError | None = None
    for attempt in range(attempts):
        try:
            return action()
        except (ValueError, MemoryError):
            # An empty or oversized window is a caller error, never transient.
            raise
        except Exception as exc:
            error = as_access_error(exc)
            if not error.is_transient:
                raise error from exc
            last = error
            if attempt < attempts - 1:
                sleep(backoff_s * (attempt + 1))
    assert last is not None
    raise last


def open_with_retry(
    path: str,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_s: float = BACKOFF_S,
    opener: Callable[[str], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Open a raster, retrying transient transport failures only."""
    if opener is None:
        import rasterio

        opener = rasterio.open
    return _retrying(lambda: opener(path), attempts=attempts, backoff_s=backoff_s, sleep=sleep)


def read_with_retry(
    dataset: Any,
    window: PixelWindow,
    *,
    band: int = 1,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_s: float = BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Read one window, retrying transient transport failures only."""
    from core.sar.windows import read_window

    return _retrying(
        lambda: read_window(dataset, window, band=band),
        attempts=attempts,
        backoff_s=backoff_s,
        sleep=sleep,
    )


@contextmanager
def open_measurement(
    product: SarProduct,
    polarisation: str = "vv",
    *,
    creds: S3Credentials | None = None,
    client: Any = None,
) -> Iterator[Any]:
    """Open one polarisation of a product for windowed reading.

    Yields an open rasterio dataset inside an environment carrying the CDSE
    credentials; both are released on exit. Locate pixels with
    :class:`core.sar.georef.SceneGeoref`, never ``dataset.transform``.
    """
    creds = creds if creds is not None else S3Credentials.from_env()
    if creds is None:
        raise S3AccessError("no_credentials")
    bucket, keys = measurement_keys(product, creds, client=client)
    key = keys.get(polarisation.lower())
    if key is None:
        found = ", ".join(sorted(keys)) or "none"
        raise S3AccessError(
            "not_found", f"no {polarisation} measurement in {product.name} (found: {found})"
        )

    with creds.rasterio_env():
        dataset = open_with_retry(f"/vsis3/{bucket}/{key}")
        try:
            yield dataset
        finally:
            dataset.close()


def probe_pixel_access() -> dict[str, Any]:
    """Read real pixels from a recent Gulf scene through the production path.

    Returns a JSON-serialisable verdict instead of raising, so the result can
    cross a process boundary intact.
    """
    started = time.perf_counter()
    try:
        import boto3  # noqa: F401
        import rasterio  # noqa: F401
    except ImportError as exc:
        return {"ok": False, "reason": "missing_dependency", "detail": str(exc)}

    creds = S3Credentials.from_env()
    if creds is None:
        return {"ok": False, "reason": "no_credentials", "detail": ""}

    try:
        from core.db.spatial import HORMUZ_BBOX
        from core.sar.catalogue import search
        from core.sar.windows import PixelWindow

        since = dt.datetime.now(dt.UTC) - dt.timedelta(days=PROBE_LOOKBACK_DAYS)
        products = [
            p for p in search(HORMUZ_BBOX, since=since) if p.is_cog and p.online and p.s3_path
        ]
        if not products:
            return {
                "ok": False,
                "reason": "not_found",
                "detail": f"no readable product over Hormuz in {PROBE_LOOKBACK_DAYS} days",
            }
        product = products[0]
        with open_measurement(product, "vv", creds=creds) as dataset:
            window = PixelWindow(
                dataset.width // 2, dataset.height // 2, PROBE_WINDOW_PX, PROBE_WINDOW_PX
            )
            pixels = read_with_retry(dataset, window)
        return {
            "ok": True,
            "reason": "ok",
            "product": product.name,
            "shape": list(pixels.shape),
            "max_dn": int(pixels.max()),
            "seconds": round(time.perf_counter() - started, 2),
        }
    except S3AccessError as exc:
        return {"ok": False, "reason": exc.reason, "detail": exc.detail}
    except Exception as exc:
        error = as_access_error(exc)
        return {"ok": False, "reason": error.reason, "detail": error.detail}


def probe_main() -> None:
    """Run :func:`probe_pixel_access` in a child process and report.

    Writes exactly one JSON line and flushes it. After a failed request the
    process may never exit on its own (see the module docstring), so the
    parent reads the line and then terminates the child rather than waiting.
    """
    verdict = probe_pixel_access()
    sys.stdout.write(json.dumps(verdict) + "\n")
    sys.stdout.flush()
