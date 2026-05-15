"""Tests for the Langfuse client.

These tests deliberately never touch the network. They inject a mock
``requests.Session`` via the :class:`LangfuseClient` constructor and
verify the happy path, the failure modes the aggregator depends on
(missing credentials, HTTP errors, network errors), and the model-id
normalisation rules used to map Langfuse identifiers onto the pricing
dict.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from masops_evaluation.langfuse_client import (
    UNKNOWN_MODEL,
    LangfuseClient,
    build_langfuse_client,
    normalize_model_id,
)


# --- normalize_model_id ----------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_family",
    [
        ("anthropic/claude-sonnet-4.6", "sonnet"),
        ("anthropic/claude-4.6-sonnet-20251101", "sonnet"),
        ("anthropic/claude-haiku-4.5", "haiku"),
        ("google/gemini-3.1-pro-preview", "gemini"),
        ("google/gemini-3.1-pro", "gemini"),
        ("ANTHROPIC/CLAUDE-SONNET-4.6", "sonnet"),
    ],
)
def test_normalize_model_id_maps_known_families(raw: str, expected_family: str) -> None:
    result = normalize_model_id(raw)
    assert result != UNKNOWN_MODEL
    assert expected_family in result.lower()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "openai/gpt-5",
        "meta/llama-3.7-405b",
        "google/gemini-3.1-flash",
    ],
)
def test_normalize_model_id_returns_unknown_for_unmapped(raw: str) -> None:
    assert normalize_model_id(raw) == UNKNOWN_MODEL


# --- LangfuseClient --------------------------------------------------------

def _make_response(status: int = 200, json_body: Any = None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.text = text
    response.json.return_value = {} if json_body is None else json_body
    return response


def _make_client(session: MagicMock, *, public: str = "pk", secret: str = "sk") -> LangfuseClient:
    return LangfuseClient(
        base_url="http://example:3000",
        public_key=public,
        secret_key=secret,
        project_id="masops-project",
        session=session,
    )


def test_get_usage_in_window_aggregates_per_model() -> None:
    body = {
        "data": [
            {
                "model": "anthropic/claude-sonnet-4.6",
                "inputTokens": 1000,
                "outputTokens": 500,
                "totalTokens": 1500,
                "count": 3,
            },
            {
                "model": "google/gemini-3.1-pro-preview",
                "inputTokens": 2000,
                "outputTokens": 800,
                "totalTokens": 2800,
                "count": 5,
            },
            # Second row for sonnet — must sum into the same bucket.
            {
                "model": "anthropic/claude-sonnet-4.6",
                "inputTokens": 200,
                "outputTokens": 100,
                "totalTokens": 300,
                "count": 1,
            },
        ]
    }
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(200, body)

    client = _make_client(session)
    result = client.get_usage_in_window(datetime(2026, 1, 1), datetime(2026, 1, 2))

    assert len(result) == 2
    by_model = {row["model"]: row for row in result}
    sonnet = by_model["anthropic/claude-sonnet-4.6"]
    assert sonnet["total_input_tokens"] == 1200
    assert sonnet["total_output_tokens"] == 600
    assert sonnet["total_tokens"] == 1800
    assert sonnet["observation_count"] == 4
    gemini = by_model["google/gemini-3.1-pro-preview"]
    assert gemini["total_tokens"] == 2800
    assert gemini["observation_count"] == 5

    session.get.assert_called_once()
    called_url = session.get.call_args[0][0]
    assert called_url == "http://example:3000/api/public/metrics/observations"
    call_kwargs = session.get.call_args.kwargs
    assert call_kwargs["auth"] == ("pk", "sk")
    assert call_kwargs["params"]["groupBy"] == "model"
    assert call_kwargs["params"]["fromTimestamp"] == "2026-01-01T00:00:00"


def test_get_usage_in_window_handles_alt_field_names() -> None:
    """The parser must accept ``usage.input``/``usage.output`` style payloads too."""
    body = {
        "data": [
            {
                "providedModelName": "anthropic/claude-sonnet-4.6",
                "usage": {"input": 100, "output": 50, "total": 150},
                "observationCount": 2,
            },
        ]
    }
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(200, body)

    result = _make_client(session).get_usage_in_window(
        datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == [
        {
            "model": "anthropic/claude-sonnet-4.6",
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "total_tokens": 150,
            "observation_count": 2,
        }
    ]


def test_get_usage_in_window_returns_empty_when_credentials_missing() -> None:
    session = MagicMock(spec=requests.Session)
    client = _make_client(session, public="", secret="")
    result = client.get_usage_in_window(datetime(2026, 1, 1), datetime(2026, 1, 2))
    assert result == []
    session.get.assert_not_called()


def test_get_usage_in_window_returns_empty_on_401() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(401, text="unauthorized")
    result = _make_client(session).get_usage_in_window(
        datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == []


def test_get_usage_in_window_returns_empty_on_network_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("refused")
    result = _make_client(session).get_usage_in_window(
        datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == []


def test_get_usage_in_window_returns_empty_on_non_json_body() -> None:
    session = MagicMock(spec=requests.Session)
    response = _make_response(200)
    response.json.side_effect = ValueError("not json")
    session.get.return_value = response
    result = _make_client(session).get_usage_in_window(
        datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == []


def test_get_total_tokens_in_window_wraps_aggregation() -> None:
    body = {
        "data": [
            {"model": "anthropic/claude-sonnet-4.6", "totalTokens": 1500},
            {"model": "google/gemini-3.1-pro-preview", "totalTokens": 2800},
        ]
    }
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(200, body)
    result = _make_client(session).get_total_tokens_in_window(
        datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == {
        "anthropic/claude-sonnet-4.6": 1500,
        "google/gemini-3.1-pro-preview": 2800,
    }


# --- build_langfuse_client -------------------------------------------------

def test_build_langfuse_client_returns_none_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from masops_evaluation import langfuse_client as mod
    fake = MagicMock()
    fake.has_langfuse_credentials.return_value = False
    monkeypatch.setattr(mod, "get_settings", lambda: fake)
    assert build_langfuse_client() is None


def test_build_langfuse_client_returns_none_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from masops_evaluation import langfuse_client as mod
    fake = MagicMock()
    fake.has_langfuse_credentials.return_value = True
    fake.effective_langfuse_url.return_value = ""
    monkeypatch.setattr(mod, "get_settings", lambda: fake)
    assert build_langfuse_client() is None


def test_build_langfuse_client_constructs_client_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from masops_evaluation import langfuse_client as mod
    fake = MagicMock()
    fake.has_langfuse_credentials.return_value = True
    fake.effective_langfuse_url.return_value = "http://10.0.1.42:3000"
    fake.langfuse_public_key = "pk"
    fake.langfuse_secret_key = "sk"
    fake.langfuse_project_id = "masops-project"
    monkeypatch.setattr(mod, "get_settings", lambda: fake)
    client = build_langfuse_client()
    assert client is not None
    assert client.base_url == "http://10.0.1.42:3000"
    assert client.auth == ("pk", "sk")
    assert client.project_id == "masops-project"
