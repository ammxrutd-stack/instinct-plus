"""YouTube Data API v3 wrapper: the pipeline's only network boundary.

Every other stage takes a client object exposing ``get(endpoint, **params)``
and ``units``, so the rest of the pipeline is exercised in tests against a
fake with no key and no network. This is the one module that knows about
HTTP status codes, retries and quota.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from config.schema import PROJECT_ROOT

API_URL = "https://www.googleapis.com/youtube/v3"

# leaves headroom under the 10,000 daily cap for manual exploration on the same key
DEFAULT_BUDGET = 9_000
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 30

# 403 reasons that mean the daily quota is spent, as opposed to the key being wrong
QUOTA_REASONS = ("quotaExceeded", "dailyLimitExceeded")


class CollectionError(RuntimeError):
    """A condition the run cannot recover from: wrong key, API not enabled, bad request.

    Raised instead of SystemExit so the stages stay testable; the CLI turns it
    into exit code 1.
    """


class QuotaExceeded(RuntimeError):
    """The next call would exceed the unit budget, or the API says the daily quota is spent."""


def load_api_key(env_path: Optional[Path] = None) -> str:
    """YOUTUBE_API_KEY from the environment, after loading PROJECT_ROOT/.env."""
    load_dotenv(env_path or PROJECT_ROOT / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise CollectionError(
            "YOUTUBE_API_KEY is not set.\n"
            "  cp .env.example .env    then paste your key into it.\n"
            "  Get one at https://console.cloud.google.com -> enable\n"
            "  'YouTube Data API v3' -> Credentials -> Create API key."
        )
    return key


def _error_details(response) -> tuple:
    """(reason, message) from an API error body, tolerating non-JSON bodies."""
    try:
        body = response.json().get("error", {})
    except ValueError:
        return "", (response.text or "")[:300]
    errors = body.get("errors") or [{}]
    return errors[0].get("reason", ""), body.get("message", "")


class Client:
    """Counts quota units against a budget and retries transient failures.

    Every list endpoint the pipeline uses costs 1 unit per call, so units are
    counted per HTTP request made, retries included. The budget is checked
    before each request so a run stops short of the cap, never past it.
    """

    def __init__(self, key, budget=DEFAULT_BUDGET, session=None, sleep=time.sleep):
        self.key = key
        self.budget = budget
        self.units = 0
        self.session = session or requests.Session()
        self.sleep = sleep  # injectable so tests do not wait on backoff

    def get(self, endpoint: str, **params) -> Optional[dict]:
        """One API call. Returns the JSON payload, or None after exhausting retries."""
        params = {name: value for name, value in params.items() if value is not None}
        params["key"] = self.key
        url = f"{API_URL}/{endpoint}"

        for attempt in range(MAX_ATTEMPTS):
            if self.units + 1 > self.budget:
                raise QuotaExceeded(
                    f"budget of {self.budget} units reached after {self.units} spent"
                )
            if attempt:
                self.sleep(2 ** (attempt - 1))

            try:
                response = self.session.get(url, params=params, timeout=TIMEOUT_SECONDS)
            except (requests.Timeout, requests.ConnectionError) as error:
                # the server may have processed it before the connection dropped,
                # so count the unit rather than risk undercounting
                self.units += 1
                print(f"  ! {endpoint}: {type(error).__name__} on attempt {attempt + 1}",
                      file=sys.stderr)
                continue

            self.units += 1
            status = response.status_code
            if status == 200:
                return response.json()
            if status in (400, 401, 403):
                self._raise_fatal(status, response)
            if status == 404:
                # a missing playlist or channel; retrying will not make it appear
                reason, _ = _error_details(response)
                print(f"  ! {endpoint}: 404 {reason}".rstrip(), file=sys.stderr)
                return None
            print(f"  ! {endpoint}: HTTP {status} on attempt {attempt + 1}", file=sys.stderr)

        print(f"  ! giving up on {endpoint} after {MAX_ATTEMPTS} attempts", file=sys.stderr)
        return None

    @staticmethod
    def _raise_fatal(status, response):
        reason, message = _error_details(response)
        if status == 403 and reason in QUOTA_REASONS:
            raise QuotaExceeded(f"API returned 403 {reason}")
        if status == 403:
            raise CollectionError(
                f"403 {reason}: the key is wrong or the YouTube Data API is not "
                f"enabled on its project.\n{message}"
            )
        raise CollectionError(f"{status} {reason}: {message}".replace(" :", ":"))
