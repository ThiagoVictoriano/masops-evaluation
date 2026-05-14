"""CLI: orchestrate the sequential evaluation of a rodada (run).

A rodada is a named set of ``(instance, repetition, source)`` executions.
For each execution we:

1. Resolve a candidate patch — either the SWE-bench gold patch or a
   synthetic mutation produced by :mod:`masops_evaluation.mutations`.
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
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
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
from masops_evaluation.mutations import (
    MUTATION_TYPES,
    MutationType,
    generate_synthetic_mutation,
    source_label_for_mutation,
)
from masops_evaluation.schemas import (
    AlternativeLabel,
    EvalMetadata,
    EvaluationRequest,
    EvaluationResponse,
    ExecutionRecord,
    PatchSource,
    PatchSourceClass,
    PullRequestPayload,
    RepositoryPayload,
    classify_confusion,
)
from masops_evaluation.select_instances import _bucket_for

logger = logging.getLogger(__name__)
console = Console()


_ALLOWED_SOURCE_CLASSES: tuple[str, ...] = ("gold", "synthetic")

# Default cap on the number of selected instances actually consumed by a
# rodada. Aligned with the budget-constrained reference configuration
# (~$20 of API credit). Override via ``--max-cases``.
DEFAULT_MAX_CASES: int = 10

# A MAS-Ops error matching this pattern halts the rodada immediately —
# continuing would only burn more budget. Covers HTTP 402/429 plus the
# common natural-language signals that an API quota or billing limit was
# hit.
_BUDGET_ERROR_RE = re.compile(
    r"\b(402|429)\b|\b(billing|credit|quota)\b",
    re.IGNORECASE,
)


def _is_budget_error(exc: BaseException) -> bool:
    """Return ``True`` when an exception message looks budget-related."""
    return bool(_BUDGET_ERROR_RE.search(str(exc)))


# --- Dataset access --------------------------------------------------------

def _load_dataset_by_id() -> dict[str, dict[str, Any]]:
    """Return a mapping ``instance_id -> row`` for the configured dataset."""
    from datasets import load_dataset

    settings = get_settings()
    console.log(f"[run] loading dataset {settings.swebench_dataset}")
    dataset = load_dataset(settings.swebench_dataset, split="test")
    return {row["instance_id"]: dict(row) for row in dataset}


# --- Patch source scheduling ----------------------------------------------

def _parse_patch_sources(raw: str) -> list[PatchSourceClass]:
    """Parse the ``--patch-sources`` CLI value into a validated list.

    Args:
        raw: Comma-separated string, e.g. ``"gold,synthetic"``.

    Returns:
        Ordered list of source classes. Order is preserved so the
        round-robin schedule respects the user's intent.

    Raises:
        ValueError: if ``raw`` is empty or contains unknown classes.
    """
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("--patch-sources must list at least one source class")
    bad = [v for v in values if v not in _ALLOWED_SOURCE_CLASSES]
    if bad:
        raise ValueError(
            f"Unknown patch source class(es): {bad!r}; allowed: {_ALLOWED_SOURCE_CLASSES}"
        )
    # Cast through Any to satisfy the Literal type checker.
    return [v for v in values]  # type: ignore[return-value]


def _build_case_schedule(
    sources: list[PatchSourceClass],
    patches_per_case: int,
) -> list[PatchSourceClass]:
    """Round-robin a list of source classes up to ``patches_per_case`` slots."""
    if patches_per_case < 1:
        raise ValueError("--patches-per-case must be >= 1")
    return [sources[i % len(sources)] for i in range(patches_per_case)]


def _mutation_seed(
    rodada_id: str,
    instance_id: str,
    repetition: int,
    mutation_type: MutationType,
) -> int:
    """Derive a deterministic seed for a mutation from the rodada coordinates."""
    raw = f"{rodada_id}|{instance_id}|{repetition}|{mutation_type}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


@dataclass
class _RotationState:
    """Tracks the global mutation-type cursor across the rodada.

    The cursor advances on every attempted synthetic mutation (whether it
    succeeded or raised ``ValueError``) so the load is balanced across the
    four mutation types over the course of the rodada.
    """

    cursor: int = 0


def _resolve_synthetic_patch(
    *,
    gold_patch: str,
    rotation: _RotationState,
    used_resolved: set[str],
    rodada_id: str,
    instance_id: str,
    repetition: int,
) -> tuple[Optional[str], Optional[PatchSource], Optional[MutationType]]:
    """Try mutation types in rotation order until one yields a usable patch.

    Args:
        gold_patch: The base gold patch to mutate.
        rotation: Mutable rotation state shared across the whole rodada.
        used_resolved: Resolved source labels already consumed in the
            current ``(instance, repetition)`` — used to avoid filename
            collisions when ``--patches-per-case`` requests multiple
            synthetic slots in the same case.
        rodada_id: Rodada identifier.
        instance_id: SWE-bench instance identifier.
        repetition: Repetition index within the rodada.

    Returns:
        ``(patch, resolved_source, mutation_type)`` on success, or
        ``(None, None, None)`` when no mutation type was applicable.
    """
    attempts = 0
    while attempts < len(MUTATION_TYPES):
        mutation_type = MUTATION_TYPES[rotation.cursor % len(MUTATION_TYPES)]
        rotation.cursor += 1
        attempts += 1
        resolved = source_label_for_mutation(mutation_type)
        if resolved in used_resolved:
            continue
        seed = _mutation_seed(rodada_id, instance_id, repetition, mutation_type)
        try:
            mutated = generate_synthetic_mutation(gold_patch, mutation_type, seed)
        except ValueError as exc:
            logger.warning(
                "Mutation %s not applicable to %s rep=%d: %s",
                mutation_type,
                instance_id,
                repetition,
                exc,
            )
            continue
        return mutated, resolved, mutation_type  # type: ignore[return-value]
    return None, None, None


def _extract_gold_patch(row: dict[str, Any]) -> str:
    """Pull the gold patch out of an instance row, validating shape."""
    patch = row.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError(
            f"Gold patch missing/empty for instance {row.get('instance_id')!r}"
        )
    return patch


def _hash_patch(patch_str: str) -> str:
    """Return a short content hash for the patch, for traceability."""
    return hashlib.sha256(patch_str.encode("utf-8")).hexdigest()[:16]


# --- Record persistence ---------------------------------------------------

def _result_path(
    results_dir: Path,
    instance_id: str,
    repetition: int,
    source: PatchSource,
) -> Path:
    """Return the canonical path for a per-execution record."""
    return results_dir / f"{instance_id}__rep-{repetition}__src-{source}.json"


def _write_record(path: Path, record: ExecutionRecord) -> None:
    """Persist a record as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")


_EXECUTION_FILENAME_RE = re.compile(
    r"^(?P<instance>.+)__rep-(?P<rep>\d+)__src-(?P<src>[A-Za-z0-9_]+)\.json$"
)
_LEGACY_FILENAME_RE = re.compile(
    r"^(?P<instance>.+)__rep-(?P<rep>\d+)\.json$"
)


def _existing_executions(
    results_dir: Path,
    instance_id: str,
) -> set[tuple[int, str]]:
    """Return ``(repetition, source)`` pairs already persisted for an instance.

    Files written by older rodadas without the ``__src-`` suffix are
    treated as ``(rep, "gold")`` so resuming a legacy run remains safe.
    """
    existing: set[tuple[int, str]] = set()
    for f in results_dir.glob(f"{instance_id}__rep-*.json"):
        m = _EXECUTION_FILENAME_RE.match(f.name)
        if m:
            existing.add((int(m.group("rep")), m.group("src")))
            continue
        legacy = _LEGACY_FILENAME_RE.match(f.name)
        if legacy:
            existing.add((int(legacy.group("rep")), "gold"))
    return existing


# --- Execution -----------------------------------------------------------

def _evaluate_one(
    *,
    instance_id: str,
    repetition: int,
    row: dict[str, Any],
    candidate_patch: str,
    resolved_source: PatchSource,
    client: MasOpsClient,
    rodada_id: str,
) -> tuple[ExecutionRecord, bool]:
    """Run one ``(instance, repetition, source)`` evaluation end to end.

    Returns:
        A ``(record, abort_rodada)`` tuple. ``abort_rodada`` is ``True``
        when the MAS-Ops call failed with a budget/quota signal — the
        orchestrator should persist this record and then stop the rodada.
    """
    execution_id = (
        f"{rodada_id}-{instance_id}-rep{repetition}-{resolved_source}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    started = datetime.now(timezone.utc).isoformat()

    difficulty_bucket = _bucket_for(row.get("difficulty")) or "unknown"
    repo = row.get("repo", "")
    patch_hash = _hash_patch(candidate_patch)

    # Step 1: harness on candidate
    input_label, input_report = run_harness(
        instance_id=instance_id,
        patch_str=candidate_patch,
        run_id_suffix=f"{rodada_id}-{instance_id}-{resolved_source}-input-rep{repetition}",
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
            title=f"Eval: {instance_id} [{resolved_source}]",
            body=row.get("problem_statement", "") or "",
        ),
        repository=RepositoryPayload(full_name=repo),
    )

    response: Optional[EvaluationResponse] = None
    error_message: Optional[str] = None
    abort_rodada = False
    try:
        response = client.submit_evaluation(request)
    except MasOpsClientError as exc:
        error_message = f"MAS-Ops call failed: {exc}"
        logger.error(error_message)
        if _is_budget_error(exc):
            abort_rodada = True
    except Exception as exc:  # noqa: BLE001 - defensive: never abort the rodada
        error_message = f"Unexpected error talking to MAS-Ops: {exc}"
        logger.exception("Unexpected MAS-Ops failure")
        if _is_budget_error(exc):
            abort_rodada = True

    record = ExecutionRecord(
        execution_id=execution_id,
        instance_id=instance_id,
        repetition=repetition,
        difficulty=difficulty_bucket,
        repo=repo,
        candidate_patch_source=resolved_source,
        candidate_patch_hash=patch_hash,
        input_label=input_label,
        harness_input_report_path=str(input_report),
        timestamp_start=started,
    )

    if response is None:
        record.error_message = error_message
        record.timestamp_end = datetime.now(timezone.utc).isoformat()
        return record, abort_rodada

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
            run_id_suffix=(
                f"{rodada_id}-{instance_id}-{resolved_source}-alt-rep{repetition}"
            ),
        )
        alt_label = alt_outcome
    record.alternative_label = alt_label
    if alt_report_path is not None:
        record.harness_alternative_report_path = str(alt_report_path)

    record.timestamp_end = datetime.now(timezone.utc).isoformat()
    return record, abort_rodada


# --- Orchestration --------------------------------------------------------

def _load_selected_instances(path: Path) -> list[str]:
    """Read ``data/selected_instances.json`` and return the consolidated list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("all_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path} has no 'all_ids' list")
    return list(ids)


def run_rodada(
    *,
    rodada_id: str,
    repetitions: int,
    patch_sources: list[PatchSourceClass],
    patches_per_case: int,
    resume: bool,
    instance_ids: Optional[list[str]],
    selection_path: Path,
    max_cases: int,
) -> dict[str, Any]:
    """Run a full rodada and return a summary dict for manifest persistence."""
    settings = get_settings()
    results_dir = settings.results_dir / rodada_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve instance set
    if instance_ids:
        full_planned_ids = list(instance_ids)
    else:
        full_planned_ids = _load_selected_instances(selection_path)

    # Apply the --max-cases cap (defaults to DEFAULT_MAX_CASES for the
    # budget-constrained reference rodada).
    if max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    planned_ids = full_planned_ids[:max_cases]
    reserve_ids = full_planned_ids[max_cases:]

    case_schedule = _build_case_schedule(patch_sources, patches_per_case)
    planned_distribution = dict(Counter(case_schedule))

    console.log(
        f"[run] rodada={rodada_id} instances={len(planned_ids)}/{len(full_planned_ids)} "
        f"(cap --max-cases={max_cases}, reserve={len(reserve_ids)}) "
        f"reps={repetitions} patches_per_case={patches_per_case} "
        f"sources={patch_sources} resume={resume}"
    )
    console.log(f"[run] case schedule (round-robin): {case_schedule}")

    # MAS-Ops pre-flight
    client = MasOpsClient()
    if not client.health_check():
        raise SystemExit(
            f"MAS-Ops health check failed at {settings.masops_health_endpoint()}. "
            "Aborting rodada."
        )

    dataset = _load_dataset_by_id()

    counts: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "ERROR": 0, "SKIPPED": 0}
    effective_distribution: Counter[str] = Counter()
    rotation = _RotationState()
    started_at = datetime.now(timezone.utc).isoformat()

    # Tracking for the manifest: cases where no synthetic mutation was
    # applicable, and whether the rodada exited early because of a
    # MAS-Ops budget / quota error.
    synthetic_unavailable: list[dict[str, Any]] = []
    aborted: bool = False
    abort_reason: Optional[str] = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_units = len(planned_ids) * repetitions * patches_per_case
        task = progress.add_task("evaluating", total=total_units)

        for instance_id in planned_ids:
            row = dataset.get(instance_id)
            if row is None:
                logger.error("Instance %s not found in dataset; skipping", instance_id)
                counts["ERROR"] += repetitions * patches_per_case
                progress.advance(task, repetitions * patches_per_case)
                continue

            existing = _existing_executions(results_dir, instance_id) if resume else set()

            for rep in range(1, repetitions + 1):
                # Pre-extract gold once per (instance, rep); skip the whole rep if missing.
                try:
                    gold_patch = _extract_gold_patch(row)
                except ValueError as exc:
                    logger.warning("Skipping %s rep=%d: %s", instance_id, rep, exc)
                    counts["ERROR"] += patches_per_case
                    progress.advance(task, patches_per_case)
                    continue

                used_resolved: set[str] = set()
                for slot_source in case_schedule:
                    # Resolve slot to a concrete (patch, resolved_source).
                    if slot_source == "gold":
                        if "gold" in used_resolved:
                            logger.warning(
                                "Duplicate gold slot for %s rep=%d; skipping",
                                instance_id,
                                rep,
                            )
                            counts["SKIPPED"] += 1
                            progress.advance(task)
                            continue
                        candidate_patch: Optional[str] = gold_patch
                        resolved_source: Optional[PatchSource] = "gold"
                    else:  # synthetic
                        candidate_patch, resolved_source, _ = _resolve_synthetic_patch(
                            gold_patch=gold_patch,
                            rotation=rotation,
                            used_resolved=used_resolved,
                            rodada_id=rodada_id,
                            instance_id=instance_id,
                            repetition=rep,
                        )
                        if candidate_patch is None or resolved_source is None:
                            logger.warning(
                                "No applicable synthetic mutation for %s rep=%d; "
                                "marking as synthetic_unavailable",
                                instance_id,
                                rep,
                            )
                            synthetic_unavailable.append(
                                {"instance_id": instance_id, "repetition": rep}
                            )
                            counts["SKIPPED"] += 1
                            progress.advance(task)
                            continue

                    used_resolved.add(resolved_source)

                    if resume and (rep, resolved_source) in existing:
                        counts["SKIPPED"] += 1
                        progress.advance(task)
                        continue

                    record, abort_signal = _evaluate_one(
                        instance_id=instance_id,
                        repetition=rep,
                        row=row,
                        candidate_patch=candidate_patch,
                        resolved_source=resolved_source,
                        client=client,
                        rodada_id=rodada_id,
                    )

                    _write_record(
                        _result_path(results_dir, instance_id, rep, resolved_source),
                        record,
                    )
                    cell = record.confusion_matrix_cell or (
                        "ERROR" if record.error_message else "ERROR"
                    )
                    counts[cell] = counts.get(cell, 0) + 1
                    effective_distribution[resolved_source] += 1
                    progress.advance(task)
                    progress.update(
                        task,
                        description=(
                            f"{instance_id} rep={rep} src={resolved_source} "
                            f"TP={counts['TP']} TN={counts['TN']} "
                            f"FP={counts['FP']} FN={counts['FN']} ERR={counts['ERROR']}"
                        ),
                    )

                    if abort_signal:
                        aborted = True
                        abort_reason = (
                            record.error_message
                            or "MAS-Ops reported a budget/quota error"
                        )
                        logger.error(
                            "Budget/quota error detected — aborting rodada to avoid "
                            "further API charges. Last record: %s",
                            record.execution_id,
                        )
                        break
                if aborted:
                    break
            if aborted:
                break

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
            "patches_per_case": patches_per_case,
            "patch_sources": patch_sources,
            "case_schedule": case_schedule,
            "max_cases": max_cases,
            "resume": resume,
            "instance_ids_override": instance_ids,
            "selection_path": str(selection_path),
            "dataset": settings.swebench_dataset,
            "masops_endpoint": settings.masops_eval_endpoint(),
        },
        "planned_instances": planned_ids,
        "reserve_instances": reserve_ids,
        "planned_distribution": planned_distribution,
        "effective_distribution": dict(effective_distribution),
        "synthetic_unavailable": synthetic_unavailable,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "counts": counts,
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if aborted:
        raise SystemExit(
            f"Rodada {rodada_id} aborted: {abort_reason}. "
            f"See manifest at {results_dir / 'manifest.json'} for the trail."
        )
    return manifest


# --- CLI ------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential MAS-Ops evaluation rodada against SWE-bench Verified. "
            "Each (instance, repetition, source) tuple produces one execution record."
        ),
    )
    parser.add_argument("--rodada-id", required=True, help="Stable identifier for this rodada.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per instance (default: 3).",
    )
    parser.add_argument(
        "--patches-per-case",
        type=int,
        default=2,
        help=(
            "Patches evaluated per (instance, repetition). "
            "Combined with --patch-sources to build a round-robin schedule "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--patch-sources",
        type=str,
        default="gold,synthetic",
        help=(
            "Comma-separated patch source classes (allowed: gold, synthetic). "
            "Default 'gold,synthetic' yields one of each per case."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip (instance, repetition, source) triples that already have a "
            "result file under results/{rodada_id}/."
        ),
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
    parser.add_argument(
        "--max-cases",
        type=int,
        default=DEFAULT_MAX_CASES,
        help=(
            "Cap the number of instances actually evaluated; the rest of the "
            "selection is kept as an overflow reserve recorded in the manifest "
            f"(default: {DEFAULT_MAX_CASES})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    try:
        sources = _parse_patch_sources(args.patch_sources)
        run_rodada(
            rodada_id=args.rodada_id,
            repetitions=args.repetitions,
            patch_sources=sources,
            patches_per_case=args.patches_per_case,
            resume=args.resume,
            instance_ids=args.instance_ids,
            selection_path=args.selection_path,
            max_cases=args.max_cases,
        )
    except SystemExit:
        raise
    except Exception:
        logger.exception("Rodada failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
