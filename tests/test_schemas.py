"""Schema-level sanity tests.

These tests intentionally do not exercise any network or filesystem paths.
They only check that the Pydantic models we expose to MAS-Ops accept the
documented payloads and reject obvious malformations.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from masops_evaluation.schemas import (
    AlternativeMetadata,
    EvalMetadata,
    EvaluationRequest,
    EvaluationResponse,
    ExecutionMetadata,
    PullRequestPayload,
    RepositoryPayload,
    classify_confusion,
)


# --- EvaluationRequest -----------------------------------------------------

def _valid_request_payload() -> dict[str, object]:
    return {
        "_eval_metadata": {
            "instance_id": "django__django-11848",
            "candidate_patch": "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@\n-x\n+y\n",
            "base_commit": "f4e93919" + "a" * 32,
            "repo": "django/django",
        },
        "pull_request": {
            "number": 1,
            "title": "Eval: django__django-11848",
            "body": "Some failing behavior in HTTP parsing.",
        },
        "repository": {"full_name": "django/django"},
    }


def test_evaluation_request_accepts_documented_payload() -> None:
    request = EvaluationRequest.model_validate(_valid_request_payload())
    assert request.eval_metadata.instance_id == "django__django-11848"
    assert request.pull_request.number == 1
    assert request.repository.full_name == "django/django"


def test_evaluation_request_round_trip_uses_alias() -> None:
    """``model_dump(by_alias=True)`` must emit the underscore-prefixed key."""
    request = EvaluationRequest.model_validate(_valid_request_payload())
    dumped = request.model_dump(by_alias=True)
    assert "_eval_metadata" in dumped
    assert "eval_metadata" not in dumped


def test_evaluation_request_rejects_missing_eval_metadata() -> None:
    payload = _valid_request_payload()
    del payload["_eval_metadata"]
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(payload)


def test_evaluation_request_rejects_missing_repo() -> None:
    payload = _valid_request_payload()
    del payload["repository"]
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(payload)


def test_eval_metadata_requires_instance_id() -> None:
    with pytest.raises(ValidationError):
        EvalMetadata.model_validate(
            {
                "candidate_patch": "x",
                "base_commit": "y",
                "repo": "django/django",
            }
        )


def test_pull_request_number_must_be_int() -> None:
    with pytest.raises(ValidationError):
        PullRequestPayload.model_validate({"number": "abc", "title": "t", "body": "b"})


def test_repository_payload_round_trip() -> None:
    payload = RepositoryPayload(full_name="django/django")
    assert payload.model_dump() == {"full_name": "django/django"}


# --- EvaluationResponse ---------------------------------------------------

def _valid_response_payload() -> dict[str, object]:
    return {
        "decision": "approve",
        "decided_by": "detective",
        "justification": "Looks good.",
        "alternative_patch": None,
        "alternative_metadata": {"guardian_iterations": 0, "guardian_final_verdict": None},
        "execution_metadata": {
            "total_tokens": 1234,
            "tokens_by_agent": {"detective": 1000, "fixer": 234},
            "agents_invoked": ["detective"],
            "duration_seconds": 12.5,
        },
    }


def test_evaluation_response_accepts_approve() -> None:
    response = EvaluationResponse.model_validate(_valid_response_payload())
    assert response.decision == "approve"
    assert response.execution_metadata.total_tokens == 1234


def test_evaluation_response_accepts_reject_with_alternative() -> None:
    payload = _valid_response_payload()
    payload["decision"] = "reject"
    payload["decided_by"] = "fixer_guardian_loop"
    payload["alternative_patch"] = "diff --git a/x b/x\n"
    payload["alternative_metadata"] = {
        "guardian_iterations": 2,
        "guardian_final_verdict": "fix accepted",
    }
    response = EvaluationResponse.model_validate(payload)
    assert response.alternative_patch is not None
    assert response.alternative_metadata.guardian_iterations == 2


def test_evaluation_response_defaults_empty_tokens_by_agent() -> None:
    payload = _valid_response_payload()
    payload["execution_metadata"] = {  # type: ignore[assignment]
        "total_tokens": 0,
        "tokens_by_agent": {},
        "agents_invoked": [],
        "duration_seconds": 0.0,
    }
    response = EvaluationResponse.model_validate(payload)
    assert response.execution_metadata.tokens_by_agent == {}


def test_evaluation_response_rejects_unknown_decision() -> None:
    payload = _valid_response_payload()
    payload["decision"] = "maybe"
    with pytest.raises(ValidationError):
        EvaluationResponse.model_validate(payload)


def test_evaluation_response_rejects_unknown_decided_by() -> None:
    payload = _valid_response_payload()
    payload["decided_by"] = "communicator"
    with pytest.raises(ValidationError):
        EvaluationResponse.model_validate(payload)


def test_execution_metadata_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        ExecutionMetadata.model_validate(
            {
                "total_tokens": -1,
                "tokens_by_agent": {},
                "agents_invoked": [],
                "duration_seconds": 0.0,
            }
        )


def test_alternative_metadata_defaults() -> None:
    meta = AlternativeMetadata()
    assert meta.guardian_iterations == 0
    assert meta.guardian_final_verdict is None


# --- classify_confusion ---------------------------------------------------

@pytest.mark.parametrize(
    "label,decision,expected",
    [
        ("resolved", "approve", "TP"),
        ("not_resolved", "approve", "FP"),
        ("resolved", "reject", "FN"),
        ("not_resolved", "reject", "TN"),
    ],
)
def test_classify_confusion_buckets(label: str, decision: str, expected: str) -> None:
    assert classify_confusion(label, decision) == expected  # type: ignore[arg-type]


def test_classify_confusion_returns_none_for_error_label() -> None:
    assert classify_confusion("error", "approve") is None  # type: ignore[arg-type]
