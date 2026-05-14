"""HTTP client for the MAS-Ops evaluation endpoint.

The MAS-Ops instance exposes two endpoints we care about:

* ``GET  /health`` (or configurable path) — pre-flight liveness check.
* ``POST /eval/pr-review`` — synthetic PR review request.

This module wraps both with a small amount of resilience (retries on 5xx,
no retries on 4xx) and structured logging.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from masops_evaluation.config import Settings, get_settings
from masops_evaluation.schemas import EvaluationRequest, EvaluationResponse

logger = logging.getLogger(__name__)


# --- Exceptions ------------------------------------------------------------

class MasOpsClientError(RuntimeError):
    """Base class for MAS-Ops client errors."""


class MasOpsTimeoutError(MasOpsClientError):
    """The MAS-Ops endpoint did not respond within the configured timeout."""


class MasOpsInvalidPayloadError(MasOpsClientError):
    """MAS-Ops rejected the payload with a 4xx status code."""


class MasOpsServerError(MasOpsClientError):
    """MAS-Ops failed with a 5xx status code after all retries were exhausted."""


# --- Client ---------------------------------------------------------------

class MasOpsClient:
    """Synchronous HTTP client for the MAS-Ops evaluation API.

    Args:
        settings: Optional pre-built :class:`Settings`. Defaults to the
            shared cached instance.
        session: Optional pre-configured :class:`requests.Session`. A new
            one is created if not provided.
        max_retries: Number of retries on 5xx responses (in addition to the
            initial attempt). Default 3.
        backoff_base_seconds: Backoff base. Sleep on retry ``i`` is
            ``base * 2**i`` (i starts at 0 → 2s, 4s, 8s with default base 2).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        session: Optional[requests.Session] = None,
        *,
        max_retries: int = 3,
        backoff_base_seconds: float = 2.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds

    # --- Public API -------------------------------------------------------

    def health_check(self) -> bool:
        """Probe the health endpoint.

        Returns:
            ``True`` when the endpoint responds with HTTP 200, ``False``
            otherwise (including network errors and 4xx/5xx responses).
        """
        url = self._settings.masops_health_endpoint()
        try:
            response = self._session.get(url, timeout=10)
        except requests.RequestException as exc:
            logger.warning("Health check to %s failed: %s", url, exc)
            return False
        ok = response.status_code == 200
        logger.info("Health check %s -> %s", url, response.status_code)
        return ok

    def submit_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        """Submit an evaluation request and return the parsed response.

        Retries on 5xx with exponential backoff. Does not retry on 4xx —
        a malformed payload will not become well-formed on retry.

        Raises:
            MasOpsInvalidPayloadError: 4xx response (payload rejected).
            MasOpsTimeoutError: timeout exceeded.
            MasOpsServerError: 5xx persisted after all retries.
            MasOpsClientError: other network-level failures.
        """
        url = self._settings.masops_eval_endpoint()
        body = request.model_dump(by_alias=True)
        timeout = self._settings.masops_timeout

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                logger.info(
                    "POST %s (attempt %d/%d) instance=%s",
                    url,
                    attempt + 1,
                    self._max_retries + 1,
                    request.eval_metadata.instance_id,
                )
                response = self._session.post(url, json=body, timeout=timeout)
            except requests.Timeout as exc:
                logger.error("Timeout on attempt %d: %s", attempt + 1, exc)
                raise MasOpsTimeoutError(
                    f"MAS-Ops did not respond within {timeout}s"
                ) from exc
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Network error on attempt %d: %s", attempt + 1, exc)
                if attempt < self._max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise MasOpsClientError(f"Network error talking to MAS-Ops: {exc}") from exc

            if 200 <= response.status_code < 300:
                return self._parse_response(response)

            if 400 <= response.status_code < 500:
                logger.error(
                    "MAS-Ops rejected payload (status=%d): %s",
                    response.status_code,
                    response.text[:500],
                )
                raise MasOpsInvalidPayloadError(
                    f"MAS-Ops returned {response.status_code}: {response.text[:500]}"
                )

            # 5xx
            logger.warning(
                "MAS-Ops 5xx on attempt %d (status=%d): %s",
                attempt + 1,
                response.status_code,
                response.text[:500],
            )
            last_error = MasOpsServerError(
                f"MAS-Ops returned {response.status_code}: {response.text[:500]}"
            )
            if attempt < self._max_retries:
                self._sleep_backoff(attempt)
                continue
            raise last_error

        # Loop exhausted without a successful return or a raised exception.
        raise MasOpsClientError(
            f"MAS-Ops request failed after {self._max_retries + 1} attempts: {last_error}"
        )

    # --- Internals --------------------------------------------------------

    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep ``base * 2**attempt`` seconds."""
        delay = self._backoff_base * (2 ** attempt)
        logger.info("Backing off %.1fs before retry", delay)
        time.sleep(delay)

    @staticmethod
    def _parse_response(response: requests.Response) -> EvaluationResponse:
        """Parse and validate the response JSON into an :class:`EvaluationResponse`."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise MasOpsClientError(
                f"MAS-Ops response was not JSON: {response.text[:500]}"
            ) from exc
        return EvaluationResponse.model_validate(payload)
