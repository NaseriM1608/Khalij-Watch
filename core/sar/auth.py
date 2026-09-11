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
from typing import TYPE_CHECKING

from core.net import open_with_retry

if TYPE_CHECKING:  # pragma: no cover
    from rasterio.env import Env

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
        with open_with_retry(request, timeout=DEFAULT_TIMEOUT, opener=opener) as response:
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


def looks_like_oauth_client(identifier: str | None) -> bool:
    """Distinguish an OAuth client id from an account username.

    CDSE supports two machine flows. An account login is an email address; an
    OAuth client created in the dashboard is an opaque id with a separate
    secret and uses a different grant entirely. Guessing wrong produces a bare
    "invalid credentials", which is a miserable thing to debug, so the shape of
    the identifier is used to pick the right grant automatically.
    """
    return bool(identifier) and "@" not in (identifier or "")


class TokenManager:
    """Fetches and refreshes a CDSE access token on demand.

    Supports both CDSE machine flows:

    * ``password`` - an account email plus password.
    * ``client_credentials`` - an OAuth client id plus secret, created in the
      CDSE dashboard. This is the better choice for a service: it is revocable
      on its own and carries no account password.

    Set either ``CDSE_CLIENT_ID``/``CDSE_CLIENT_SECRET`` or
    ``CDSE_USERNAME``/``CDSE_PASSWORD``; the grant is chosen automatically.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        opener=None,
    ) -> None:
        # Explicit client variables win; otherwise fall back to the generic
        # pair, which may hold either kind of credential.
        client_id = os.environ.get("CDSE_CLIENT_ID")
        client_secret = os.environ.get("CDSE_CLIENT_SECRET")
        self._username = username or client_id or os.environ.get("CDSE_USERNAME")
        self._password = password or client_secret or os.environ.get("CDSE_PASSWORD")
        self._opener = opener
        self._token: Token | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self._username and self._password)

    @property
    def grant_type(self) -> str:
        return "client_credentials" if looks_like_oauth_client(self._username) else "password"

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
                "No CDSE credentials. Set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET "
                "(an OAuth client from the CDSE dashboard) or "
                "CDSE_USERNAME / CDSE_PASSWORD (an account login). Register "
                "free at https://dataspace.copernicus.eu"
            )
        if self.grant_type == "client_credentials":
            payload = {
                "client_id": self._username or "",
                "client_secret": self._password or "",
                "grant_type": "client_credentials",
            }
        else:
            payload = {
                "client_id": PUBLIC_CLIENT_ID,
                "username": self._username or "",
                "password": self._password or "",
                "grant_type": "password",
            }
        return _to_token(_post(payload, self._opener), now)

    def _refresh(self, refresh_token: str, now: dt.datetime) -> Token:
        # A client_credentials token carries no refresh token; re-authenticating
        # is both cheap and correct there.
        if self.grant_type == "client_credentials":
            return self._login(now)
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

    @property
    def endpoint_host(self) -> str:
        """The endpoint as a bare host, which is what GDAL expects."""
        return self.endpoint.removeprefix("https://").removeprefix("http://").rstrip("/")

    def gdal_config(self) -> dict[str, str]:
        """Non-secret GDAL settings for ``/vsis3/`` access to CDSE storage.

        Deliberately carries NO credentials. rasterio refuses AWS keys passed
        as GDAL options: it raises ``EnvError`` and insists they travel in a
        session object. An earlier version returned the keys from here, so the
        read path crashed the first time it met a real raster while every unit
        test passed, because no test ever opened a raster through it.
        """
        return {
            # Path-style addressing: a virtual-hosted request would prepend the
            # bucket name to the endpoint host, which does not resolve.
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
            # Read only the byte ranges actually needed. Without this GDAL
            # lists the product directory and pulls far more than the window
            # requires, which defeats the point of a COG.
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
            # Not running on EC2, so never probe the instance metadata service.
            "CPL_AWS_AUTODETECT_EC2": "NO",
            "GDAL_HTTP_TIMEOUT": "60",
            "GDAL_HTTP_MAX_RETRY": "3",
            "GDAL_HTTP_RETRY_DELAY": "2",
        }

    def rasterio_env(self) -> Env:
        """A ``rasterio.Env`` able to read from CDSE object storage.

        Credentials travel through rasterio's ``AWSSession``, the only route
        it accepts; everything else is plain GDAL configuration. Needs boto3.
        """
        from rasterio.env import Env
        from rasterio.session import AWSSession

        session = AWSSession(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=S3_REGION,
            endpoint_url=self.endpoint_host,
        )
        return Env(session=session, **self.gdal_config())
