"""Read-only HTTP client for the self-hosted Langfuse public API.

The evaluation framework instruments only the MAS-Ops agents that go
through ``llm.client.call_llm`` (detective, fixer, fixer_guardian,
executor_guardian). The Executor (Claude Agent SDK) and Communicator
(``langchain.create_agent``) bypass that path, so their token usage is
missing from :attr:`ExecutionRecord.tokens_by_agent`. Langfuse traces on
EC2-mas-ops capture *every* LLM call, which lets the aggregator report
an end-to-end cost figure that includes those agents.

The client is best-effort: any HTTP, auth, or connectivity failure is
logged at WARNING and surfaced as an empty result, so the aggregator can
fall back to the partial in-process estimate without aborting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import requests

from masops_evaluation.config import Settings, get_settings

logger = logging.getLogger(__name__)


# Canonical mapping from a coarse model family (matched heuristically against
# the Langfuse identifier) to the exact key used in
# ``aggregate_results.MODEL_PRICING_USD_PER_MTOK``. Keeping a single
# canonical key per family lets the aggregator use the existing pricing
# dict directly, without a parallel "Langfuse pricing" table.
_PRICING_CANONICAL_KEY = {
    "sonnet": "anthropic/claude-sonnet-4.6",
    "haiku": "anthropic/claude-haiku-4.5",
    "gemini_pro": "google/gemini-3.1-pro-preview",
}

UNKNOWN_MODEL = "unknown"


def normalize_model_id(langfuse_model_id: str) -> str:
    """Map a raw Langfuse model identifier to a known pricing key.

    Langfuse reports models under their OpenRouter-style identifiers,
    which include date suffixes and provider prefixes (e.g.
    ``"anthropic/claude-4.6-sonnet-20251101"``). This function
    collapses those into one of the canonical keys present in the
    aggregator's pricing dict.

    Returns ``"unknown"`` when no rule matches — the caller must keep the
    tokens in the running total but exclude them from cost summation.
    """
    if not langfuse_model_id:
        return UNKNOWN_MODEL
    lower = langfuse_model_id.lower()
    if "sonnet" in lower:
        return _PRICING_CANONICAL_KEY["sonnet"]
    if "haiku" in lower:
        return _PRICING_CANONICAL_KEY["haiku"]
    if "gemini" in lower and "pro" in lower:
        return _PRICING_CANONICAL_KEY["gemini_pro"]
    return UNKNOWN_MODEL


class LangfuseClient:
    """Read-only client for the Langfuse public API.

    Args:
        base_url: Root URL of the Langfuse instance (no trailing slash).
        public_key: Public key used as Basic Auth username.
        secret_key: Secret key used as Basic Auth password.
        project_id: Project identifier; carried for logging context.
        timeout_seconds: Per-request HTTP timeout.
        session: Optional pre-built :class:`requests.Session`.
    """

    # Path of the Langfuse usage-by-observation metric endpoint. Exposed as
    # a class attribute so a subclass can swap it if the deployed Langfuse
    # version uses a different route.
    OBSERVATIONS_METRICS_PATH = "/api/public/metrics/observations"

    def __init__(
        self,
        base_url: str,
        public_key: str,
        secret_key: str,
        project_id: str,
        *,
        timeout_seconds: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = (public_key, secret_key)
        self.project_id = project_id
        self._timeout = timeout_seconds
        self._session = session or requests.Session()

    def get_usage_in_window(
        self,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Return per-model usage aggregated across ``[start, end]``.

        Each item is a dict with:
            ``model`` (raw Langfuse identifier),
            ``total_input_tokens``,
            ``total_output_tokens``,
            ``total_tokens``,
            ``observation_count``.

        On any HTTP, auth, or parse failure the method logs at WARNING
        and returns ``[]`` — the aggregator treats that as "no Langfuse
        data" and skips the corresponding report section.
        """
        if not self.base_url:
            logger.warning("Langfuse base URL is empty; skipping usage query.")
            return []
        if not (self.auth[0] and self.auth[1]):
            logger.warning("Langfuse credentials missing; skipping usage query.")
            return []

        url = f"{self.base_url}{self.OBSERVATIONS_METRICS_PATH}"
        params = {
            "fromTimestamp": start.isoformat(),
            "toTimestamp": end.isoformat(),
            "groupBy": "model",
        }
        try:
            response = self._session.get(
                url,
                params=params,
                auth=self.auth,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Langfuse request to %s failed: %s", url, exc)
            return []

        if response.status_code != 200:
            logger.warning(
                "Langfuse returned HTTP %d for %s: %s",
                response.status_code,
                url,
                response.text[:400],
            )
            return []

        try:
            body = response.json()
        except ValueError as exc:
            logger.warning("Langfuse response was not JSON: %s", exc)
            return []

        return _parse_observations_metrics(body)

    def get_total_tokens_in_window(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, int]:
        """Return ``{model_name: total_tokens}`` for the window.

        Thin convenience wrapper over :meth:`get_usage_in_window` with
        identical failure semantics (empty dict on any error).
        """
        usage = self.get_usage_in_window(start, end)
        return {item["model"]: int(item["total_tokens"]) for item in usage}


def build_langfuse_client(settings: Optional[Settings] = None) -> Optional[LangfuseClient]:
    """Construct a :class:`LangfuseClient` from settings, or ``None`` when unusable.

    Returns ``None`` (without raising) in either of these cases:
    - the Langfuse credentials are not set;
    - the base URL cannot be derived (no explicit ``LANGFUSE_URL`` and no
      ``MASOPS_HOST`` to derive from).

    Callers should treat ``None`` as "Langfuse unavailable" and degrade
    gracefully — the aggregator does so explicitly.
    """
    s = settings or get_settings()
    if not s.has_langfuse_credentials():
        logger.info("Langfuse credentials not configured; client unavailable.")
        return None
    base_url = s.effective_langfuse_url()
    if not base_url:
        logger.warning(
            "Cannot derive Langfuse URL (LANGFUSE_URL empty and MASOPS_HOST unset)."
        )
        return None
    return LangfuseClient(
        base_url=base_url,
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        project_id=s.langfuse_project_id,
    )


# --- Response parsing -----------------------------------------------------


def _parse_observations_metrics(body: Any) -> list[dict[str, Any]]:
    """Flatten a Langfuse metrics response into our internal usage schema.

    The Langfuse response shape varies across versions (wrapped under
    ``"data"``, bare list, or paginated envelope) and field names vary too
    (``inputTokens`` vs ``promptTokens`` vs ``usage.input``). This parser
    accepts the common variants so the client survives a Langfuse upgrade
    without code changes.
    """
    items: list[Any]
    if isinstance(body, dict):
        items = body.get("data") or body.get("items") or body.get("observations") or []
    elif isinstance(body, list):
        items = body
    else:
        logger.warning("Unexpected Langfuse response type: %s", type(body).__name__)
        return []

    aggregated: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        model = (
            raw.get("model")
            or raw.get("providedModelName")
            or raw.get("modelName")
            or UNKNOWN_MODEL
        )
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        input_tokens = _coerce_int(
            raw.get("inputTokens")
            or raw.get("input_tokens")
            or raw.get("promptTokens")
            or usage.get("input")
            or usage.get("promptTokens")
        )
        output_tokens = _coerce_int(
            raw.get("outputTokens")
            or raw.get("output_tokens")
            or raw.get("completionTokens")
            or usage.get("output")
            or usage.get("completionTokens")
        )
        total_tokens = _coerce_int(
            raw.get("totalTokens")
            or raw.get("total_tokens")
            or usage.get("total")
        )
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        count = _coerce_int(
            raw.get("count")
            or raw.get("observationCount")
            or raw.get("observation_count")
            or 1
        )
        bucket = aggregated.setdefault(
            model,
            {
                "model": model,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "observation_count": 0,
            },
        )
        bucket["total_input_tokens"] += input_tokens
        bucket["total_output_tokens"] += output_tokens
        bucket["total_tokens"] += total_tokens
        bucket["observation_count"] += count
    return list(aggregated.values())


def _coerce_int(value: Any) -> int:
    """Return ``int(value)`` or 0 when value is None / non-numeric."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
