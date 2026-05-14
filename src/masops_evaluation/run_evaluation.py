"""CLI: orchestrate the sequential evaluation of a rodada (run).

A rodada is a named set of (instance, repetition) executions. For each
execution we:

1. Pick a candidate patch (currently: the SWE-bench gold patch).
2. Label that patch with the SWE-bench harness — this is ground truth.
3. POST a synthetic PR to MAS-Ops and capture the decision.
4. If MAS-Ops returns an alternative patch, label that too.
5. Persist a structured :class:`ExecutionRecord` JSON.

Execution is strictly sequential: the MAS-Ops system is known to mishandle
concurrent ``/eval/pr-review`` requests, so we never overlap them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from masops_evaluation.config import configure_logging, get_settings
from masops_evaluation.harness_client import run_harness
from masops_evaluation.masops_client import MasOpsClient, MasOpsClientError
from masops_evaluation.schemas import (
    AlternativeLabel,
    EvalMetadata,
    EvaluationRequest,
    EvaluationResponse,
    ExecutionRecord,
    PatchSource,
    PullRequestPayload,
    RepositoryPayload,
    classify_confusion,
)
from masops_evaluation.select_instances import _bucket_for

logger = logging.getLogger(__name__)
console = Console()


# --- Dataset access --------------------------------------------------------

def _load_dataset_by_id() -> dict[str, dict[str, Any]]:
    """Return a mapping ``instance_id -> row`` for the configured dataset."""
    from datasets import load_dataset

    settings = get_settings()
    console.log(f"[run] loading dataset {settings.swebench_dataset}")
    dataset = load_dataset(settings.swebench_dataset, split="test")
    return {row["instance_id"]: dict(row) for row in dataset}


# --- Patch selection -------------------------------------------------------

def _select_candidate_patch(row: dict[str, Any], source: PatchSource) -> str:
    """Pick the candidate patch for an instance row.

    The ``trajectory`` and ``synthetic`` modes are placeholders. They raise
    ``NotImplementedError`` until the data pipelines that produce them land.

    Args:
        row: Instance row from the SWE-bench dataset.
        source: Selected patch source.

    Returns:
        The diff string to evaluate.
    """
    if source == "gold":
        patch = row.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError(
                f"Gold patch missing/empty for instance {row.get('instance_id')!r}"
            )
        return patch
    if source == "trajectory":
        raise NotImplementedError(
            "Trajectory patch source is not implemented yet — "
            "wire up the failed-agent trajectory dataset before using --patch-source trajectory."
        )
    if source == "synthetic":
        raise NotImplementedError(
            "Synthetic patch mutation is not implemented yet — "
            "wire up the patch-mutation pipeline before using --patch-source synthetic."
        )
    raise ValueError(f"Unknown patch source: {source!r}")


def _hash_patch(patch_str: str) -> str:
    """Return a short content hash for the patch, for traceability."""
    return hashlib.sha256(patch_str.encode("utf-8")).hexdigest()[:16]


# --- Record persistence ---------------------------------------------------

def _result_path(results_dir: Path, instance_id: str, repetition: int) -> Path:
    """Return the canonical path for a per-execution record."""
    return results_dir / f"{instance_id}__rep-{repetition}.json"


def _write_record(path: Path, record: ExecutionRecord) -> None:
    """Persist a record as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


# --- Execution loop -------------------------------------------------------

def _evaluate_one(
    *,
    instance_id: str,
    repetition: int,
    row: dict[str, Any],
    patch_source: PatchSource,
    client: MasOpsClient,
    rodada_id: str,
) -> ExecutionRecord:
    """Run one (instance, repetition) evaluation end to end."""
    settings = get_settings()
    execution_id = f"{rodada_id}-{instance_id}-rep{repetition}-{uuid.uuid4().hex[:8]}"
    started = datetime.now(timezone.utc).isoformat()

    difficulty_bucket = _bucket_for(row.get("difficulty")) or "unknown"
    repo = row.get("repo", "")

    try:
        candidate_patch = _select_candidate_patch(row, patch_source)
    except (ValueError, NotImplementedError) as exc:
        return ExecutionRecord(
            execution_id=execution_id,
            instance_id=instance_id,
            repetition=repetition,
            difficulty=difficulty_bucket,
            repo=repo,
            candidate_patch_source=patch_source,
            candidate_patch_hash="",
            input_label="error",
            timestamp_start=started,
            timestamp_end=datetime.now(timezone.utc).isoformat(),
            error_message=f"Patch selection failed: {exc}",
        )

    patch_hash = _hash_patch(candidate_patch)

    # Step 1: harness on candidate
    input_label, input_report = run_harness(
        instance_id=instance_id,
        patch_str=candidate_patch,
        run_id_suffix=f"{rodada_id}-{instance_id}-input-rep{repetition}",
    )

    # Step 2: MAS-Ops request
    request = EvaluationRequest(
        _eval_metadata=EvalMetadata(
            instance_id=instance_id,
            candidate_patch=candidate_patch,
            base_commit=row.get("base_commit", ""),
            repo=repo,
        ),
        pull_request=PullRequestPayload(
            number=repetition,
            title=f"Eval: {instance_id}",
            body=row.get("problem_statement", "") or "",
        ),
        repository=RepositoryPayload(full_name=repo),
    )

    response: Optional[EvaluationResponse] = None
    error_message: Optional[str] = None
    try:
        response = client.submit_evaluation(request)
    except MasOpsClientError as exc:
        error_message = f"MAS-Ops call failed: {exc}"
        logger.error(error_message)
    except Exception as exc:  # noqa: BLE001 - defensive: never abort the rodada
        error_message = f"Unexpected error talking to MAS-Ops: {exc}"
        logger.exception("Unexpected MAS-Ops failure")

    record = ExecutionRecord(
        execution_id=execution_id,
        instance_id=instance_id,
        repetition=repetition,
        difficulty=difficulty_bucket,
        repo=repo,
        candidate_patch_source=patch_source,
        candidate_patch_hash=patch_hash,
        input_label=input_label,
        harness_input_report_path=str(input_report),
        timestamp_start=started,
    )

    if response is None:
        record.error_message = error_message
        record.timestamp_end = datetime.now(timezone.utc).isoformat()
        return record

    # Step 3: pull MAS-Ops decision into the record
    record.mas_ops_decision = response.decision
    record.decided_by = response.decided_by
    record.justification = response.justification
    record.alternative_patch = response.alternative_patch
    record.guardian_iterations = response.alternative_metadata.guardian_iterations
    record.agents_invoked = response.execution_metadata.agents_invoked
    record.total_tokens = response.execution_metadata.total_tokens
    record.tokens_by_agent = dict(response.execution_metadata.tokens_by_agent)
    record.duration_seconds = response.execution_metadata.duration_seconds
    record.confusion_matrix_cell = classify_confusion(input_label, response.decision)

    # Step 4: optional second harness call on the alternative
    alt_label: AlternativeLabel = "none"
    alt_report_path: Optional[Path] = None
    if response.alternative_patch and response.alternative_patch.strip():
        alt_outcome, alt_report_path = run_harness(
            instance_id=instance_id,
            patch_str=response.alternative_patch,
            run_id_suffix=f"{rodada_id}-{instance_id}-alt-rep{repetition}",
        )
        alt_label = alt_outcome  # narrows to subtype of AlternativeLabel
    record.alternative_label = alt_label
    if alt_report_path is not None:
        record.harness_alternative_report_path = str(alt_report_path)

    record.timestamp_end = datetime.now(timezone.utc).isoformat()
    _ = settings  # silence linters if settings becomes unused after edits
    return record


# --- Orchestration --------------------------------------------------------

def _load_selected_instances(path: Path) -> list[str]:
    """Read ``data/selected_instances.json`` and return the consolidated list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("all_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path} has no 'all_ids' list")
    return list(ids)


def _existing_repetitions(results_dir: Path, instance_id: str) -> set[int]:
    """Return the repetitions already persisted for an instance."""
    completed: set[int] = set()
    for f in results_dir.glob(f"{instance_id}__rep-*.json"):
        try:
            n = int(f.stem.split("rep-")[-1])
        except ValueError:
            continue
        completed.add(n)
    return completed


def run_rodada(
    *,
    rodada_id: str,
    repetitions: int,
    patch_source: PatchSource,
    resume: bool,
    instance_ids: Optional[list[str]],
    selection_path: Path,
) -> dict[str, Any]:
    """Run a full rodada and return a summary dict for manifest persistence."""
    settings = get_settings()
    results_dir = settings.results_dir / rodada_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve instance set
    if instance_ids:
        planned_ids = list(instance_ids)
    else:
        planned_ids = _load_selected_instances(selection_path)

    console.log(
        f"[run] rodada={rodada_id} instances={len(planned_ids)} reps={repetitions} "
        f"patch_source={patch_source} resume={resume}"
    )

    # MAS-Ops pre-flight
    client = MasOpsClient()
    if not client.health_check():
        raise SystemExit(
            f"MAS-Ops health check failed at {settings.masops_health_endpoint()}. "
            "Aborting rodada."
        )

    dataset = _load_dataset_by_id()

    counts: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "ERROR": 0, "SKIPPED": 0}
    started_at = datetime.now(timezone.utc).isoformat()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_units = len(planned_ids) * repetitions
        task = progress.add_task("evaluating", total=total_units)

        for instance_id in planned_ids:
            row = dataset.get(instance_id)
            if row is None:
                logger.error("Instance %s not found in dataset; skipping", instance_id)
                counts["ERROR"] += repetitions
                progress.advance(task, repetitions)
                continue

            existing = _existing_repetitions(results_dir, instance_id) if resume else set()

            for rep in range(1, repetitions + 1):
                if rep in existing:
                    counts["SKIPPED"] += 1
                    progress.advance(task)
                    continue

                record = _evaluate_one(
                    instance_id=instance_id,
                    repetition=rep,
                    row=row,
                    patch_source=patch_source,
                    client=client,
                    rodada_id=rodada_id,
                )

                _write_record(_result_path(results_dir, instance_id, rep), record)
                cell = record.confusion_matrix_cell or ("ERROR" if record.error_message else "ERROR")
                counts[cell] = counts.get(cell, 0) + 1
                progress.advance(task)
                progress.update(
                    task,
                    description=(
                        f"{instance_id} rep={rep} "
                        f"TP={counts['TP']} TN={counts['TN']} "
                        f"FP={counts['FP']} FN={counts['FN']} ERR={counts['ERROR']}"
                    ),
                )

    ended_at = datetime.now(timezone.utc).isoformat()

    # Summary table
    table = Table(title=f"Rodada {rodada_id} summary")
    table.add_column("Cell", style="cyan")
    table.add_column("Count", justify="right")
    for cell in ("TP", "TN", "FP", "FN", "ERROR", "SKIPPED"):
        table.add_row(cell, str(counts.get(cell, 0)))
    console.print(table)

    manifest = {
        "rodada_id": rodada_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "config": {
            "repetitions": repetitions,
            "patch_source": patch_source,
            "resume": resume,
            "instance_ids_override": instance_ids,
            "selection_path": str(selection_path),
            "dataset": settings.swebench_dataset,
            "masops_endpoint": settings.masops_eval_endpoint(),
        },
        "planned_instances": planned_ids,
        "counts": counts,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --- CLI ------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sequential MAS-Ops evaluation rodada against SWE-bench Verified.",
    )
    parser.add_argument("--rodada-id", required=True, help="Stable identifier for this rodada.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per instance (default: 3).",
    )
    parser.add_argument(
        "--patch-source",
        choices=("gold", "trajectory", "synthetic"),
        default="gold",
        help="Source for the candidate patch (default: gold).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (instance, repetition) pairs that already have a result file.",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        default=None,
        help="Optional explicit list of instance IDs to evaluate (overrides selection file).",
    )
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=Path("data/selected_instances.json"),
        help="Path to the selection JSON produced by select-instances.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    try:
        run_rodada(
            rodada_id=args.rodada_id,
            repetitions=args.repetitions,
            patch_source=args.patch_source,
            resume=args.resume,
            instance_ids=args.instance_ids,
            selection_path=args.selection_path,
        )
    except SystemExit:
        raise
    except Exception:
        logger.exception("Rodada failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
