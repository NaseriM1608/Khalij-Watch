"""Tests for CDSE authentication.

No live credentials are needed: the identity service is faked, so token
lifecycle and the GDAL configuration are both verified offline.
"""

from __future__ import annotations

import datetime as dt
import io
import json

import pytest

from core.sar.auth import (
    EXPIRY_MARGIN_S,
    AuthError,
    S3Credentials,
    Token,
    TokenManager,
    looks_like_oauth_client,
)

NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.UTC)
# An account login is an email; an OAuth client id is opaque. The grant is
# chosen from that difference, so the tests must use realistic values.
ACCOUNT = "analyst@example.com"
CLIENT_ID = "sh-2f1c9a44-0e17-4b5b-9d3e-77c2a1b45e90"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Never let the developer's real credentials leak into these tests."""
    for name in (
        "CDSE_CLIENT_ID",
        "CDSE_CLIENT_SECRET",
        "CDSE_USERNAME",
        "CDSE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


def fake_identity(*responses):
    """An opener returning scripted identity-service payloads."""
    queue = list(responses)
    calls: list[dict] = []

    def opener(request, timeout=None):  # noqa: ARG001
        from urllib.parse import parse_qs

        calls.append(parse_qs(request.data.decode()))
        payload = queue.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return _Response(json.dumps(payload).encode())

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_token_is_valid_before_expiry():
    token = Token("abc", None, NOW + dt.timedelta(minutes=10))

    assert token.is_valid(NOW)


def test_token_expires_early_by_the_safety_margin():
    """A token must never lapse mid-request."""
    token = Token("abc", None, NOW + dt.timedelta(seconds=EXPIRY_MARGIN_S - 1))

    assert not token.is_valid(NOW)


def test_an_email_selects_the_password_grant():
    opener = fake_identity({"access_token": "AT", "refresh_token": "RT", "expires_in": 600})
    manager = TokenManager(ACCOUNT, "pass", opener=opener)

    assert manager.grant_type == "password"
    assert manager.token(now=NOW) == "AT"
    assert opener.calls[0]["grant_type"] == ["password"]


def test_an_opaque_id_selects_the_client_credentials_grant():
    """An OAuth client from the CDSE dashboard, not an account login."""
    opener = fake_identity({"access_token": "AT", "expires_in": 600})
    manager = TokenManager(CLIENT_ID, "secret", opener=opener)

    assert manager.grant_type == "client_credentials"
    assert manager.token(now=NOW) == "AT"
    assert opener.calls[0]["grant_type"] == ["client_credentials"]
    assert opener.calls[0]["client_id"] == [CLIENT_ID]


def test_client_credentials_reauthenticates_instead_of_refreshing():
    """That grant issues no refresh token, so a fresh login is correct."""
    opener = fake_identity(
        {"access_token": "AT1", "expires_in": 600},
        {"access_token": "AT2", "expires_in": 600},
    )
    manager = TokenManager(CLIENT_ID, "secret", opener=opener)

    manager.token(now=NOW)
    second = manager.token(now=NOW + dt.timedelta(minutes=11))

    assert second == "AT2"
    assert opener.calls[1]["grant_type"] == ["client_credentials"]


def test_explicit_client_variables_win_over_the_generic_pair(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("CDSE_USERNAME", ACCOUNT)
    monkeypatch.setenv("CDSE_PASSWORD", "pw")

    assert TokenManager().grant_type == "client_credentials"


def test_oauth_client_detection():
    assert looks_like_oauth_client(CLIENT_ID)
    assert not looks_like_oauth_client(ACCOUNT)
    assert not looks_like_oauth_client(None)


def test_a_valid_token_is_reused_not_refetched():
    opener = fake_identity({"access_token": "AT", "expires_in": 600})
    manager = TokenManager(ACCOUNT, "pass", opener=opener)

    manager.token(now=NOW)
    manager.token(now=NOW + dt.timedelta(seconds=30))

    assert len(opener.calls) == 1


def test_an_expired_token_is_refreshed():
    opener = fake_identity(
        {"access_token": "AT1", "refresh_token": "RT", "expires_in": 600},
        {"access_token": "AT2", "refresh_token": "RT2", "expires_in": 600},
    )
    manager = TokenManager(ACCOUNT, "pass", opener=opener)

    manager.token(now=NOW)
    second = manager.token(now=NOW + dt.timedelta(minutes=11))

    assert second == "AT2"
    assert opener.calls[1]["grant_type"] == ["refresh_token"]


def test_a_dead_refresh_token_falls_back_to_full_login():
    """The refresh token has its own expiry; re-authenticating is correct."""
    import urllib.error

    opener = fake_identity(
        {"access_token": "AT1", "refresh_token": "RT", "expires_in": 600},
        urllib.error.HTTPError("u", 400, "bad", {}, None),
        {"access_token": "AT3", "expires_in": 600},
    )
    manager = TokenManager(ACCOUNT, "pass", opener=opener)

    manager.token(now=NOW)
    third = manager.token(now=NOW + dt.timedelta(minutes=11))

    assert third == "AT3"
    assert opener.calls[2]["grant_type"] == ["password"]


def test_missing_credentials_give_an_actionable_error():
    manager = TokenManager(None, None)

    with pytest.raises(AuthError, match="CDSE_CLIENT_ID"):
        manager.token(now=NOW)


def test_is_configured_reflects_credentials():
    assert not TokenManager(None, None).is_configured
    assert TokenManager(ACCOUNT, "p").is_configured


def test_bad_credentials_are_reported_clearly():
    import urllib.error

    opener = fake_identity(urllib.error.HTTPError("u", 401, "no", {}, None))
    manager = TokenManager(ACCOUNT, "wrong", opener=opener)

    with pytest.raises(AuthError, match="invalid credentials"):
        manager.token(now=NOW)


def test_response_without_a_token_is_rejected():
    opener = fake_identity({"error": "nope"})

    with pytest.raises(AuthError, match="no access_token"):
        TokenManager(ACCOUNT, "p", opener=opener).token(now=NOW)


# --------------------------------------------------------------------------
# S3
# --------------------------------------------------------------------------


def test_s3_credentials_read_from_environment(monkeypatch):
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "AK")
    monkeypatch.setenv("CDSE_S3_SECRET_KEY", "SK")

    creds = S3Credentials.from_env()

    assert creds is not None
    assert creds.access_key == "AK"


def test_s3_credentials_absent_returns_none(monkeypatch):
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_KEY", raising=False)

    assert S3Credentials.from_env() is None


def test_gdal_options_strip_the_scheme_from_the_endpoint():
    """GDAL wants a bare host in AWS_S3_ENDPOINT, not a URL."""
    options = S3Credentials("AK", "SK").gdal_options()

    assert options["AWS_S3_ENDPOINT"] == "eodata.dataspace.copernicus.eu"
    assert "https://" not in options["AWS_S3_ENDPOINT"]


def test_gdal_options_avoid_listing_the_bucket():
    """Without this GDAL pulls far more than the window needs from a COG."""
    options = S3Credentials("AK", "SK").gdal_options()

    assert options["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert options["AWS_VIRTUAL_HOSTING"] == "FALSE"
