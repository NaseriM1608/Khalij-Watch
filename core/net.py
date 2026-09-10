"""HTTP helpers shared by the CDSE clients.

DNS resolution on this host has proven intermittent - the same name resolves
fine seconds later. A single failure must therefore not be reported as "this
service is unavailable", so every request retries transient network faults
while leaving real HTTP responses alone for the caller to interpret.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.5
DEFAULT_TIMEOUT = 90

# A default Python user-agent is sometimes refused by CDN front-ends.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def open_with_retry(
    request: urllib.request.Request,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    opener: Any = None,
    sleep: Any = time.sleep,
) -> Any:
    """Open a request, retrying transient DNS and network failures.

    An :class:`urllib.error.HTTPError` is re-raised immediately: the server
    answered, and what that answer means is the caller's business. Only
    genuine transport failures are retried.
    """
    open_fn = opener or urllib.request.urlopen
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return open_fn(request, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                sleep(backoff_s * (attempt + 1))
    assert last is not None
    raise last
