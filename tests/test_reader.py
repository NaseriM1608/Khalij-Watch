"""Tests for the object-storage reader.

The error strings below are the exact messages the live CDSE endpoint and
GDAL produced on 2026-09-11, so classification is checked against what the
service really says rather than what documentation suggests it might.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.sar.auth import S3Credentials
from core.sar.reader import (
    REMEDIES,
    S3AccessError,
    as_access_error,
    classify_s3_error,
    measurement_keys,
    open_with_retry,
    read_with_retry,
    split_s3_path,
)
from core.sar.windows import PixelWindow

UNKNOWN_KEY = (
    '<?xml version="1.0" encoding="UTF-8"?><Error><Code>InvalidAccessKeyId</Code>'
    "<RequestId>5049ba90-8194-9b09-98d2-a05467ce69a1</RequestId><HostId></HostId></Error>"
)
WRONG_SECRET = (
    "SignatureDoesNotMatch: The request signature we calculated does not match "
    "the signature you provided"
)
DNS_FAILURE = "CURL error: Could not resolve host: eodata.dataspace.copernicus.eu"

PREFIX = (
    "Sentinel-1/SAR/IW_GRDH_1S-COG/2026/09/10/"
    "S1D_IW_GRDH_1SDV_20260910T020640_20260910T020705_004509_00860C_1202_COG.SAFE"
)
S3_PATH = "/eodata/" + PREFIX
VV_KEY = (
    PREFIX + "/measurement/s1d-iw-grd-vv-20260910t020640-20260910t020705-004509-00860c-001-cog.tiff"
)
VH_KEY = (
    PREFIX + "/measurement/s1d-iw-grd-vh-20260910t020640-20260910t020705-004509-00860c-002-cog.tiff"
)

CREDS = S3Credentials("AKIDEXAMPLE", "SECRETEXAMPLE")


def product(s3_path: str | None = S3_PATH) -> SimpleNamespace:
    return SimpleNamespace(name="S1D_TEST_COG.SAFE", s3_path=s3_path)


class FakeClient:
    """A stand-in for boto3 that records what it was asked for."""

    def __init__(self, contents=None, error: Exception | None = None) -> None:
        self.contents = contents or []
        self.error = error
        self.requests: list[dict] = []

    def list_objects_v2(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"Contents": [{"Key": key} for key in self.contents]}


class ClientError(Exception):
    """Mimics botocore's ClientError, which carries the S3 code in ``response``."""

    def __init__(self, code: str) -> None:
        super().__init__(f"An error occurred ({code})")
        self.response = {"Error": {"Code": code}}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (UNKNOWN_KEY, "unknown_access_key"),
        (WRONG_SECRET, "wrong_secret"),
        (DNS_FAILURE, "network"),
        ("HTTP response code: 503", "network"),
        ("HTTP response code: 403", "access_denied"),
        ("<Code>NoSuchKey</Code>", "not_found"),
        ("something no one anticipated", "other"),
    ],
)
def test_real_error_messages_are_classified(message, reason):
    assert classify_s3_error(message) == reason


def test_a_credential_error_wins_over_its_http_status():
    """A wrong secret also arrives as a 403; the code is the useful part."""
    assert classify_s3_error("HTTP response code: 403 " + WRONG_SECRET) == "wrong_secret"


def test_only_network_failures_are_transient():
    assert S3AccessError("network").is_transient
    assert not S3AccessError("wrong_secret").is_transient
    assert not S3AccessError("unknown_access_key").is_transient


def test_the_error_message_carries_the_remedy_and_detail():
    error = S3AccessError("wrong_secret", "SignatureDoesNotMatch")

    assert REMEDIES["wrong_secret"] in str(error)
    assert "SignatureDoesNotMatch" in str(error)


def test_an_unknown_key_remedy_mentions_expiry():
    """CDSE keys expire on a date chosen at creation, which is easy to forget."""
    assert "expiry" in REMEDIES["unknown_access_key"]


def test_a_boto3_style_error_is_classified_by_its_code():
    error = as_access_error(ClientError("InvalidAccessKeyId"))

    assert error.reason == "unknown_access_key"


def test_an_access_error_passes_through_unchanged():
    original = S3AccessError("wrong_secret")

    assert as_access_error(original) is original


# --------------------------------------------------------------------------
# Locating measurement rasters
# --------------------------------------------------------------------------


def test_split_s3_path_separates_bucket_and_prefix():
    assert split_s3_path(S3_PATH) == ("eodata", PREFIX)


@pytest.mark.parametrize("path", ["", "/", "/eodata", "/eodata/"])
def test_split_s3_path_rejects_paths_without_a_prefix(path):
    with pytest.raises(ValueError):
        split_s3_path(path)


def test_measurement_keys_are_found_by_polarisation():
    client = FakeClient([VH_KEY, VV_KEY])

    bucket, keys = measurement_keys(product(), CREDS, client=client)

    assert bucket == "eodata"
    assert keys == {"vv": VV_KEY, "vh": VH_KEY}
    assert client.requests[0]["Prefix"] == PREFIX + "/measurement/"


def test_non_raster_files_are_ignored():
    client = FakeClient([VV_KEY, PREFIX + "/measurement/s1d-iw-grd-vv-notes.xml"])

    _, keys = measurement_keys(product(), CREDS, client=client)

    assert keys == {"vv": VV_KEY}


def test_a_product_without_an_s3_path_is_reported():
    with pytest.raises(S3AccessError) as raised:
        measurement_keys(product(s3_path=None), CREDS, client=FakeClient())

    assert raised.value.reason == "not_found"


def test_a_rejected_listing_names_the_credential_problem():
    client = FakeClient(error=ClientError("SignatureDoesNotMatch"))

    with pytest.raises(S3AccessError) as raised:
        measurement_keys(product(), CREDS, client=client)

    assert raised.value.reason == "wrong_secret"


# --------------------------------------------------------------------------
# Retrying
# --------------------------------------------------------------------------


def flaky(failures: list[Exception], result: object):
    """A callable that raises each queued failure in turn, then succeeds."""
    calls = {"count": 0}

    def call(*_args, **_kwargs):
        calls["count"] += 1
        if failures:
            raise failures.pop(0)
        return result

    call.calls = calls  # type: ignore[attr-defined]
    return call


def test_an_intermittent_dns_failure_is_retried():
    sleeps: list[float] = []
    opener = flaky([OSError(DNS_FAILURE), OSError(DNS_FAILURE)], "dataset")

    dataset = open_with_retry("/vsis3/eodata/x.tiff", opener=opener, sleep=sleeps.append)

    assert dataset == "dataset"
    assert opener.calls["count"] == 3
    assert sleeps == [2.0, 4.0]


def test_a_wrong_secret_is_not_retried():
    """Retrying a bad credential only delays the message the operator needs."""
    sleeps: list[float] = []
    opener = flaky([OSError(WRONG_SECRET)], "dataset")

    with pytest.raises(S3AccessError) as raised:
        open_with_retry("/vsis3/eodata/x.tiff", opener=opener, sleep=sleeps.append)

    assert raised.value.reason == "wrong_secret"
    assert opener.calls["count"] == 1
    assert sleeps == []


def test_retries_give_up_after_the_last_attempt():
    opener = flaky([OSError(DNS_FAILURE)] * 5, "dataset")

    with pytest.raises(S3AccessError) as raised:
        open_with_retry("/vsis3/eodata/x.tiff", opener=opener, sleep=lambda _s: None)

    assert raised.value.reason == "network"
    assert opener.calls["count"] == 3


def test_a_transient_read_failure_is_retried():
    pytest.importorskip("rasterio")
    pixels = np.ones((4, 4), dtype=np.uint16)
    dataset = SimpleNamespace(dtypes=["uint16"], read=flaky([OSError(DNS_FAILURE)], pixels))

    result = read_with_retry(dataset, PixelWindow(0, 0, 4, 4), sleep=lambda _s: None)

    assert result is pixels


def test_a_caller_error_is_never_retried_or_disguised():
    """An empty window is a bug in the caller, not a network problem."""
    dataset = SimpleNamespace(dtypes=["uint16"], read=flaky([], None))

    with pytest.raises(ValueError, match="empty window"):
        read_with_retry(dataset, PixelWindow(0, 0, 0, 0), sleep=lambda _s: None)


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------


def _s3_keys_from_dotenv(monkeypatch) -> bool:
    """Load only the S3 keys from .env, scoped to one test via monkeypatch."""
    from pathlib import Path

    dotenv = Path(__file__).resolve().parents[1] / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().startswith("CDSE_S3_"):
                monkeypatch.setenv(key.strip(), value.strip())
    return S3Credentials.from_env() is not None


@pytest.mark.network
def test_real_pixels_are_read_through_the_production_reader(monkeypatch):
    """The test that would have caught the original crash.

    Opens a live Sentinel-1 scene through exactly the path the pipeline uses,
    including handing real credentials to rasterio. Skipped without S3 keys.
    """
    pytest.importorskip("rasterio")
    pytest.importorskip("boto3")
    if not _s3_keys_from_dotenv(monkeypatch):
        pytest.skip("CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY not set")
    from core.sar.reader import probe_pixel_access

    verdict = probe_pixel_access()

    assert verdict["ok"], verdict
    assert verdict["max_dn"] > 0
