"""Copernicus Data Space authentication.

Two separate credential sets, for two different jobs:

* **OAuth username/password** - identifies you to the OData and download APIs.
  The access token lives ~10 minutes and the refresh token ~60, so anything
  long-running must refresh rather than assume a token stays valid.
* **S3 access/secret key** - generated in the CDSE dashboard, used by GDAL to
  make byte-range reads against ``/eodata`` object storage. This is the path
  that lets an AOI window be read without downloading a whole scene.

Catalogue *search* needs neither (see :mod:`core.sar.catalogue`), which is why
scene discovery works before any of this is configured.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
PUBLIC_CLIENT_ID = "cdse-public"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
S3_REGION = "default"

# Refresh this far before nominal expiry, so a token cannot lapse mid-request.
EXPIRY_MARGIN_S = 60
DEFAULT_TIMEOUT = 45


class AuthError(RuntimeError):
    """Credentials are missing, rejected, or the identity service failed."""


@dataclass(frozen=True)
class Token:
    access_token: str
    refresh_token: str | None
    expires_at: dt.datetime

    def is_valid(self, now: dt.datetime | None = None) -> bool:
        now = now or dt.datetime.now(dt.UTC)
        return now < self.expires_at - dt.timedelta(seconds=EXPIRY_MARGIN_S)


def _post(payload: dict[str, str], opener=None) -> dict:
    body = urllib.parse.urlencode(payload).encode()
    request = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=DEFAULT_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = "invalid credentials" if exc.code in (400, 401) else f"HTTP {exc.code}"
        raise AuthError(f"CDSE token request failed: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuthError(f"CDSE identity service unreachable: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise AuthError(f"CDSE returned invalid JSON: {exc}") from exc


def _to_token(payload: dict, now: dt.datetime) -> Token:
    access = payload.get("access_token")
    if not access:
        raise AuthError("CDSE response contained no access_token")
    expires_in = int(payload.get("expires_in", 600))
    return Token(
        access_token=access,
        refresh_token=payload.get("refresh_token"),
        expires_at=now + dt.timedelta(seconds=expires_in),
    )


class TokenManager:
    """Fetches and refreshes a CDSE access token on demand."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        opener=None,
    ) -> None:
        self._username = username or os.environ.get("CDSE_USERNAME")
        self._password = password or os.environ.get("CDSE_PASSWORD")
        self._opener = opener
        self._token: Token | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self._username and self._password)

    def token(self, *, now: dt.datetime | None = None) -> str:
        """Return a valid access token, refreshing or re-authenticating."""
        now = now or dt.datetime.now(dt.UTC)
        if self._token and self._token.is_valid(now):
            return self._token.access_token
        if self._token and self._token.refresh_token:
            try:
                self._token = self._refresh(self._token.refresh_token, now)
                return self._token.access_token
            except AuthError:
                # The refresh token has its own, longer expiry; falling back to
                # a full login is correct rather than surfacing this.
                pass
        self._token = self._login(now)
        return self._token.access_token

    def _login(self, now: dt.datetime) -> Token:
        if not self.is_configured:
            raise AuthError(
                "CDSE_USERNAME / CDSE_PASSWORD are not set. Register free at "
                "https://dataspace.copernicus.eu and add them to .env"
            )
        return _to_token(
            _post(
                {
                    "client_id": PUBLIC_CLIENT_ID,
                    "username": self._username or "",
                    "password": self._password or "",
                    "grant_type": "password",
                },
                self._opener,
            ),
            now,
        )

    def _refresh(self, refresh_token: str, now: dt.datetime) -> Token:
        return _to_token(
            _post(
                {
                    "client_id": PUBLIC_CLIENT_ID,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                self._opener,
            ),
            now,
        )


@dataclass(frozen=True)
class S3Credentials:
    """Object-storage keys for byte-range reads against /eodata."""

    access_key: str
    secret_key: str
    endpoint: str = S3_ENDPOINT

    @classmethod
    def from_env(cls) -> S3Credentials | None:
        access = os.environ.get("CDSE_S3_ACCESS_KEY")
        secret = os.environ.get("CDSE_S3_SECRET_KEY")
        if not access or not secret:
            return None
        return cls(
            access_key=access,
            secret_key=secret,
            endpoint=os.environ.get("CDSE_S3_ENDPOINT", S3_ENDPOINT),
        )

    def gdal_options(self) -> dict[str, str]:
        """GDAL configuration for ``/vsis3/`` access to CDSE object storage."""
        host = self.endpoint.removeprefix("https://").removeprefix("http://")
        return {
            "AWS_S3_ENDPOINT": host,
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_REGION": S3_REGION,
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
            # Read only the byte ranges actually needed. Without this GDAL will
            # happily pull far more of the object than the window requires,
            # which defeats the entire point of using a COG.
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
        }
