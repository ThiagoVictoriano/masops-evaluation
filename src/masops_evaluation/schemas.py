"""Pydantic schemas and persistence dataclasses for the evaluation framework.

The pydantic models mirror the contract with the MAS-Ops ``/eval/pr-review``
endpoint exactly (see README for the contract reference). The
:class:`ExecutionRecord` dataclass captures everything we persist per
execution to ``results/{rodada_id}/{instance_id}__rep-{n}.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- Type aliases -----------------------------------------------------------

Decision = Literal["approve", "reject"]
DecidedBy = Literal["detective", "fixer_guardian_loop"]
PatchLabel = Literal["resolved", "not_resolved", "error"]
AlternativeLabel = Literal["resolved", "not_resolved", "error", "none"]
ConfusionCell = Literal["TP", "TN", "FP", "FN"]
PatchSource = Literal["gold", "trajectory", "synthetic"]


# --- Request schemas --------------------------------------------------------

class EvalMetadata(BaseModel):
    """Metadata block consumed by MAS-Ops to bootstrap the eval flow."""

    instance_id: str = Field(..., description="SWE-bench instance identifier.")
    candidate_patch: str = Field(..., description="Diff string under evaluation.")
    base_commit: str = Field(..., description="Repo SHA the patch applies on top of.")
    repo: str = Field(..., description="GitHub-style ``owner/name`` repo identifier.")


class PullRequestPayload(BaseModel):
    """Synthetic PR header forwarded to MAS-Ops."""

    number: int = Field(..., description="Synthetic PR number.")
    title: str = Field(..., description="Synthetic PR title.")
    body: str = Field(..., description="Issue problem statement as PR body.")


class RepositoryPayload(BaseModel):
    """Repository identification block."""

    full_name: str = Field(..., description="GitHub-style ``owner/name`` repo identifier.")


class EvaluationRequest(BaseModel):
    """Full POST body sent to ``/eval/pr-review``.

    The field name ``_eval_metadata`` is non-standard for Python but mandated
    by the MAS-Ops contract, hence the alias.
    """

    model_config = ConfigDict(populate_by_name=True)

    eval_metadata: EvalMetadata = Field(..., alias="_eval_metadata")
    pull_request: PullRequestPayload
    repository: RepositoryPayload


# --- Response schemas ------------------------------------------------------

class AlternativeMetadata(BaseModel):
    """Metadata associated with an alternative patch proposal."""

    guardian_iterations: int = Field(default=0, ge=0)
    guardian_final_verdict: Optional[str] = None


class ExecutionMetadata(BaseModel):
    """Runtime metadata reported by MAS-Ops for a single evaluation."""

    total_tokens: int = Field(default=0, ge=0)
    tokens_by_agent: dict[str, int] = Field(
        default_factory=dict,
        description="Per-agent token usage (only agents using llm.client.call_llm).",
    )
    agents_invoked: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0, ge=0.0)


class EvaluationResponse(BaseModel):
    """Full response body returned by ``/eval/pr-review``."""

    decision: Decision
    decided_by: DecidedBy
    justification: str
    alternative_patch: Optional[str] = None
    alternative_metadata: AlternativeMetadata = Field(default_factory=AlternativeMetadata)
    execution_metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata)


# --- Persistence record ----------------------------------------------------

@dataclass
class ExecutionRecord:
    """One row in the per-rodada results directory.

    A single instance/repetition pair produces exactly one record. Errors
    populate ``error_message`` but the record is still written so the rodada
    has a complete trail of attempts.
    """

    # Identity ---------------------------------------------------------------
    execution_id: str
    instance_id: str
    repetition: int
    difficulty: str
    repo: str

    # Candidate patch -------------------------------------------------------
    candidate_patch_source: PatchSource
    candidate_patch_hash: str

    # Ground truth (harness over candidate) ---------------------------------
    input_label: PatchLabel

    # MAS-Ops decision ------------------------------------------------------
    mas_ops_decision: Optional[Decision] = None
    decided_by: Optional[DecidedBy] = None
    justification: Optional[str] = None

    # Alternative patch -----------------------------------------------------
    alternative_patch: Optional[str] = None
    alternative_label: AlternativeLabel = "none"
    guardian_iterations: int = 0

    # Agents / cost ---------------------------------------------------------
    agents_invoked: list[str] = field(default_factory=list)
    total_tokens: int = 0
    tokens_by_agent: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    # Evaluation classification --------------------------------------------
    confusion_matrix_cell: Optional[ConfusionCell] = None

    # Harness artifact paths -----------------------------------------------
    harness_input_report_path: Optional[str] = None
    harness_alternative_report_path: Optional[str] = None

    # Timing ---------------------------------------------------------------
    timestamp_start: str = ""
    timestamp_end: str = ""

    # Errors ---------------------------------------------------------------
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


def classify_confusion(input_label: PatchLabel, decision: Decision) -> Optional[ConfusionCell]:
    """Map (ground truth, MAS-Ops decision) to a confusion-matrix cell.

    Convention: the model is treated as a binary classifier where ``approve``
    is the positive class. ``input_label="error"`` is unclassifiable and
    returns ``None`` — callers should leave ``confusion_matrix_cell`` unset.

    Args:
        input_label: Harness label on the candidate patch (ground truth).
        decision: MAS-Ops decision on the synthetic PR.

    Returns:
        The matrix cell, or ``None`` when the input label is unusable.
    """
    if input_label == "error":
        return None
    if input_label == "resolved" and decision == "approve":
        return "TP"
    if input_label == "not_resolved" and decision == "approve":
        return "FP"
    if input_label == "resolved" and decision == "reject":
        return "FN"
    if input_label == "not_resolved" and decision == "reject":
        return "TN"
    return None
