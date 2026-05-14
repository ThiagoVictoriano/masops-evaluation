"""CLI: aggregate per-execution JSONs into a consolidated report.

For a given ``rodada_id``, this command:

1. Loads every ``ExecutionRecord`` JSON from ``results/{rodada_id}/``.
2. Computes deterministic metrics (confusion matrix, F1, PRRR, PFNRR, etc.).
3. Renders matplotlib charts to ``consolidated/{rodada_id}/charts/``.
4. Writes ``report.md`` and ``all_executions.csv`` to
   ``consolidated/{rodada_id}/``.

The qualitative narrative sections (executive summary, observations,
notable cases) are produced by Anthropic LLM stubs whose prompts are
marked TODO for later prompt-engineering work.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rich.console import Console
from rich.table import Table

from masops_evaluation.config import configure_logging, get_settings
from masops_evaluation.schemas import ConfusionCell, ExecutionRecord

logger = logging.getLogger(__name__)
console = Console()


# --- Pricing and expected model assignment -------------------------------

# USD per 1M tokens, as ``(input_price, output_price)``. Used by
# :func:`estimate_cost` to translate aggregate token counts into a rough
# dollar figure. Prices are best-effort references; update as upstream
# pricing changes.
MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "anthropic/claude-sonnet-4.6": (3.0, 15.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "google/gemini-3.1-pro-preview": (2.0, 12.0),
    "google/gemini-3.1-pro": (2.0, 12.0),
}

# Expected agent→model assignment for the reference rodada. The
# evaluation framework cannot verify this from response payloads (MAS-Ops
# does not echo back the model id per agent), so this mapping is the
# *documented* configuration; the report flags it as expected, not
# verified.
EXPECTED_AGENT_MODELS: dict[str, str] = {
    "detective": "google/gemini-3.1-pro-preview",
    "fixer": "google/gemini-3.1-pro-preview",
    "fixer_guardian": "anthropic/claude-sonnet-4.6",
    "executor": "anthropic/claude-sonnet-4.6",
    "executor_guardian": "anthropic/claude-sonnet-4.6",
    # Communicator is expected to be off (EVAL_INVOKE_COMMUNICATOR=false) in
    # budget-constrained rodadas; it has no associated pricing entry here.
}


def estimate_cost(
    tokens_by_agent_total: dict[str, int],
    assumed_input_share: float = 0.5,
) -> dict[str, Any]:
    """Estimate USD cost from aggregate per-agent token counts.

    MAS-Ops reports a single ``total_tokens`` per agent, without an
    input/output breakdown. The cost estimate assumes ``assumed_input_share``
    of those tokens are input and the rest are output. Tweak via the
    parameter if your workload skews differently.

    Args:
        tokens_by_agent_total: Sum of tokens per agent across the rodada.
        assumed_input_share: Fraction of tokens treated as input
            (0.0 = all output, 1.0 = all input). Default 0.5.

    Returns:
        Dict with ``total_usd_estimate``, ``per_agent`` breakdowns, and
        ``assumed_input_share`` echoed back for transparency.
    """
    per_agent: dict[str, dict[str, Any]] = {}
    total_usd = 0.0
    for agent, tokens in tokens_by_agent_total.items():
        model = EXPECTED_AGENT_MODELS.get(agent, "unknown")
        pricing = MODEL_PRICING_USD_PER_MTOK.get(model)
        if pricing is None:
            per_agent[agent] = {
                "model": model,
                "tokens": tokens,
                "usd_estimate": None,
            }
            continue
        in_price, out_price = pricing
        avg_price = in_price * assumed_input_share + out_price * (1 - assumed_input_share)
        usd = tokens * avg_price / 1_000_000
        per_agent[agent] = {
            "model": model,
            "tokens": tokens,
            "input_price_per_mtok": in_price,
            "output_price_per_mtok": out_price,
            "usd_estimate": usd,
        }
        total_usd += usd
    return {
        "total_usd_estimate": total_usd,
        "per_agent": per_agent,
        "assumed_input_share": assumed_input_share,
    }


def _infer_communicator_status(records: list[ExecutionRecord]) -> str:
    """Heuristically infer whether the Communicator was active in this rodada."""
    for r in records:
        if any("communicator" in (a or "").lower() for a in r.agents_invoked):
            return "active (observed in agents_invoked)"
    return "skipped or disabled (no 'communicator' invocation observed)"


def _load_manifest(results_dir: Path) -> dict[str, Any]:
    """Load ``manifest.json`` from a rodada directory; return empty dict if missing."""
    path = results_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse manifest at %s: %s", path, exc)
        return {}


# --- Loading -------------------------------------------------------------

def _load_records(results_dir: Path) -> list[ExecutionRecord]:
    """Load every execution record JSON in a rodada directory."""
    records: list[ExecutionRecord] = []
    for path in sorted(results_dir.glob("*__rep-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed record %s", path)
            continue
        try:
            records.append(ExecutionRecord(**data))
        except TypeError as exc:
            logger.warning("Skipping record with unexpected fields %s: %s", path, exc)
    return records


def _split_records(
    records: list[ExecutionRecord],
) -> tuple[list[ExecutionRecord], list[ExecutionRecord]]:
    """Partition records into ``(valid, errored)``.

    A record is "errored" when ``error_message`` is set — typically a
    failed MAS-Ops call. Errored records are excluded from quantitative
    metrics but are reported as a tally so the audit trail stays
    complete.
    """
    valid = [r for r in records if not r.error_message]
    errored = [r for r in records if r.error_message]
    return valid, errored


def _summarise_errors(errored: list[ExecutionRecord]) -> dict[str, Any]:
    """Group error records by a short reason fingerprint."""
    reason_counts: Counter[str] = Counter()
    for r in errored:
        msg = (r.error_message or "").strip()
        # Take the first 80 chars of the first line as the fingerprint.
        first_line = msg.splitlines()[0] if msg else "<empty>"
        reason_counts[first_line[:80]] += 1
    return {
        "total": len(errored),
        "by_reason": dict(reason_counts),
        "samples": [
            {
                "instance_id": r.instance_id,
                "repetition": r.repetition,
                "source": r.candidate_patch_source,
                "error_message": r.error_message,
            }
            for r in errored[:5]
        ],
    }


def _source_class(source: str) -> str:
    """Map a concrete patch source label to its high-level class."""
    return "gold" if source == "gold" else "synthetic"


def _detect_incomplete_pairs(
    records: list[ExecutionRecord],
    expected_classes: set[str],
) -> list[dict[str, Any]]:
    """Identify ``(instance, repetition)`` keys missing one of the expected source classes.

    Args:
        records: Successful records only (errors should already be filtered).
        expected_classes: Source classes the rodada was configured to run
            (e.g. ``{"gold", "synthetic"}``).
    """
    by_pair: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in records:
        by_pair[(r.instance_id, r.repetition)].add(_source_class(r.candidate_patch_source))
    incomplete: list[dict[str, Any]] = []
    for (inst, rep), present in sorted(by_pair.items()):
        missing = expected_classes - present
        if missing:
            incomplete.append(
                {
                    "instance_id": inst,
                    "repetition": rep,
                    "present": sorted(present),
                    "missing": sorted(missing),
                }
            )
    return incomplete


def _filter_complete_pairs(
    records: list[ExecutionRecord],
    expected_classes: set[str],
) -> list[ExecutionRecord]:
    """Return only records whose ``(instance, repetition)`` covers every expected class."""
    by_pair: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in records:
        by_pair[(r.instance_id, r.repetition)].add(_source_class(r.candidate_patch_source))
    keep_keys = {key for key, classes in by_pair.items() if expected_classes.issubset(classes)}
    return [r for r in records if (r.instance_id, r.repetition) in keep_keys]


# --- Metric computation --------------------------------------------------

def _confusion_counts(records: Iterable[ExecutionRecord]) -> dict[ConfusionCell, int]:
    """Tally TP/TN/FP/FN from a record iterable."""
    counts: dict[ConfusionCell, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in records:
        cell = r.confusion_matrix_cell
        if cell in counts:
            counts[cell] += 1
    return counts


def _safe_div(num: float, den: float) -> float:
    """Division that returns 0.0 when the denominator is zero (NaN-safe)."""
    return num / den if den else 0.0


def _classification_metrics(counts: dict[ConfusionCell, int]) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 from a confusion-matrix dict."""
    tp = counts.get("TP", 0)
    tn = counts.get("TN", 0)
    fp = counts.get("FP", 0)
    fn = counts.get("FN", 0)
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "support": total,
    }


def _remediation_rates(records: list[ExecutionRecord]) -> dict[str, float]:
    """Compute PRRR and PFNRR with alternative_label=='none' excluded."""
    tn_with_alt = [r for r in records if r.confusion_matrix_cell == "TN" and r.alternative_label != "none"]
    fn_with_alt = [r for r in records if r.confusion_matrix_cell == "FN" and r.alternative_label != "none"]
    prrr = _safe_div(
        sum(1 for r in tn_with_alt if r.alternative_label == "resolved"),
        len(tn_with_alt),
    )
    pfnrr = _safe_div(
        sum(1 for r in fn_with_alt if r.alternative_label == "resolved"),
        len(fn_with_alt),
    )
    return {
        "prrr": prrr,
        "prrr_support": float(len(tn_with_alt)),
        "pfnrr": pfnrr,
        "pfnrr_support": float(len(fn_with_alt)),
    }


def _aggregate_costs(records: list[ExecutionRecord]) -> dict[str, Any]:
    """Compute aggregate cost/runtime metrics (totals and per-agent sums)."""
    tokens = [r.total_tokens for r in records]
    durations = [r.duration_seconds for r in records]
    iterations = [r.guardian_iterations for r in records]

    per_agent: Counter[str] = Counter()
    for r in records:
        for agent, value in r.tokens_by_agent.items():
            per_agent[agent] += int(value)

    return {
        "total_tokens_mean": statistics.fmean(tokens) if tokens else 0.0,
        "total_tokens_sum": sum(tokens),
        "duration_seconds_mean": statistics.fmean(durations) if durations else 0.0,
        "guardian_iterations_mean": statistics.fmean(iterations) if iterations else 0.0,
        "tokens_by_agent_total": dict(per_agent),
    }


def _by_difficulty(records: list[ExecutionRecord]) -> dict[str, dict[str, Any]]:
    """Group metrics by difficulty bucket."""
    grouped: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for r in records:
        grouped[r.difficulty or "unknown"].append(r)
    return {
        bucket: {
            "n": len(group),
            "confusion": _confusion_counts(group),
            "classification": _classification_metrics(_confusion_counts(group)),
            "remediation": _remediation_rates(group),
            "cost": _aggregate_costs(group),
        }
        for bucket, group in sorted(grouped.items())
    }


def _by_source(records: list[ExecutionRecord]) -> dict[str, dict[str, Any]]:
    """Group metrics by ``candidate_patch_source``."""
    grouped: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for r in records:
        grouped[r.candidate_patch_source or "unknown"].append(r)
    return {
        source: {
            "n": len(group),
            "confusion": _confusion_counts(group),
            "classification": _classification_metrics(_confusion_counts(group)),
            "remediation": _remediation_rates(group),
            "cost": _aggregate_costs(group),
        }
        for source, group in sorted(grouped.items())
    }


def _mutation_detection_stats(records: list[ExecutionRecord]) -> dict[str, dict[str, Any]]:
    """For each synthetic mutation type, summarise detection quality.

    For mutations whose harness ground truth is ``not_resolved`` (i.e. the
    mutation successfully broke the patch), the detection rate is the
    fraction MAS-Ops correctly rejected. Mutations that still resolved are
    tracked separately because they exercise scope/style discipline rather
    than correctness detection.
    """
    grouped: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for r in records:
        source = r.candidate_patch_source or ""
        if source.startswith("synthetic_"):
            grouped[source].append(r)

    out: dict[str, dict[str, Any]] = {}
    for source, group in sorted(grouped.items()):
        n_total = len(group)
        broken = [r for r in group if r.input_label == "not_resolved"]
        still_resolved = [r for r in group if r.input_label == "resolved"]
        errored = [r for r in group if r.input_label == "error"]
        n_correctly_rejected = sum(1 for r in broken if r.mas_ops_decision == "reject")
        n_resolved_and_approved = sum(
            1 for r in still_resolved if r.mas_ops_decision == "approve"
        )
        out[source] = {
            "n_total": n_total,
            "n_ground_truth_not_resolved": len(broken),
            "n_ground_truth_resolved": len(still_resolved),
            "n_ground_truth_error": len(errored),
            "n_correctly_rejected": n_correctly_rejected,
            "detection_rate": _safe_div(n_correctly_rejected, len(broken)),
            "n_resolved_and_approved": n_resolved_and_approved,
            "approve_rate_when_still_resolved": _safe_div(
                n_resolved_and_approved, len(still_resolved)
            ),
        }
    return out


def _repetition_variability(records: list[ExecutionRecord]) -> dict[str, float]:
    """Quantify how much repetitions of the same instance disagree."""
    by_instance: dict[str, list[ExecutionRecord]] = defaultdict(list)
    for r in records:
        by_instance[r.instance_id].append(r)

    decision_disagreement = 0
    token_stdevs: list[float] = []
    for group in by_instance.values():
        if len(group) < 2:
            continue
        decisions = {r.mas_ops_decision for r in group if r.mas_ops_decision}
        if len(decisions) > 1:
            decision_disagreement += 1
        token_values = [r.total_tokens for r in group if r.total_tokens]
        if len(token_values) >= 2:
            token_stdevs.append(statistics.stdev(token_values))

    n_groups = sum(1 for g in by_instance.values() if len(g) >= 2)
    return {
        "instances_with_repetitions": float(n_groups),
        "instances_with_decision_disagreement": float(decision_disagreement),
        "decision_disagreement_rate": _safe_div(decision_disagreement, n_groups),
        "mean_token_stdev_within_instance": (
            statistics.fmean(token_stdevs) if token_stdevs else 0.0
        ),
    }


def compute_metrics(records: list[ExecutionRecord]) -> dict[str, Any]:
    """Compute the full metrics bundle that feeds both charts and report."""
    overall_counts = _confusion_counts(records)
    return {
        "n_records": len(records),
        "n_errors": sum(1 for r in records if r.error_message),
        "confusion": overall_counts,
        "classification": _classification_metrics(overall_counts),
        "remediation": _remediation_rates(records),
        "cost": _aggregate_costs(records),
        "by_difficulty": _by_difficulty(records),
        "by_source": _by_source(records),
        "mutation_detection": _mutation_detection_stats(records),
        "variability": _repetition_variability(records),
    }


# --- Charts --------------------------------------------------------------

def _render_charts(records: list[ExecutionRecord], metrics: dict[str, Any], out_dir: Path) -> None:
    """Render the four standard charts under ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Confusion-matrix heatmap ---
    counts = metrics["confusion"]
    matrix = [
        [counts.get("TP", 0), counts.get("FN", 0)],
        [counts.get("FP", 0), counts.get("TN", 0)],
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred approve", "Pred reject"])
    ax.set_yticks([0, 1], labels=["GT resolved", "GT not_resolved"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Tokens by difficulty (boxplot) ---
    by_diff: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_diff[r.difficulty or "unknown"].append(r.total_tokens)
    buckets = sorted(by_diff.keys())
    if buckets:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([by_diff[b] for b in buckets], labels=buckets)
        ax.set_ylabel("total_tokens")
        ax.set_title("Tokens by difficulty")
        fig.tight_layout()
        fig.savefig(out_dir / "tokens_by_difficulty.png", dpi=150)
        plt.close(fig)

    # --- Accuracy by difficulty (bar) ---
    diff_metrics = metrics.get("by_difficulty", {})
    if diff_metrics:
        labels = list(diff_metrics.keys())
        accuracies = [diff_metrics[b]["classification"]["accuracy"] for b in labels]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, accuracies)
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy by difficulty")
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_by_difficulty.png", dpi=150)
        plt.close(fig)

    # --- Guardian iterations distribution ---
    iter_values = [r.guardian_iterations for r in records]
    if iter_values:
        fig, ax = plt.subplots(figsize=(6, 4))
        max_i = max(iter_values)
        bins = range(0, max(max_i + 2, 2))
        ax.hist(iter_values, bins=list(bins), edgecolor="black")
        ax.set_xlabel("guardian_iterations")
        ax.set_ylabel("count")
        ax.set_title("Guardian iterations distribution")
        fig.tight_layout()
        fig.savefig(out_dir / "guardian_iterations.png", dpi=150)
        plt.close(fig)


# --- CSV export ----------------------------------------------------------

def _records_to_dataframe(records: list[ExecutionRecord]) -> pd.DataFrame:
    """Flatten records into a pandas DataFrame for CSV export."""
    rows: list[dict[str, Any]] = []
    for r in records:
        row = r.to_dict()
        # tokens_by_agent is a dict; expand into columns for ergonomics.
        for agent, value in (row.pop("tokens_by_agent", {}) or {}).items():
            row[f"tokens_by_agent__{agent}"] = value
        row["agents_invoked"] = ",".join(row.get("agents_invoked") or [])
        rows.append(row)
    return pd.DataFrame(rows)


# --- LLM-backed narrative stubs -----------------------------------------

def _anthropic_client() -> Any:
    """Lazily build an Anthropic client; raise if key missing."""
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot produce narrative sections.")
    return Anthropic(api_key=settings.anthropic_api_key)


def generate_executive_summary(metrics: dict[str, Any]) -> str:
    """Produce a 1-paragraph executive summary of the rodada.

    TODO: prompt engineering. Target ~150 words covering accuracy, F1,
    remediation rates, and any difficulty-stratified surprises.
    """
    # TODO(prompt-engineering): replace stub with real prompt+model call.
    try:
        client = _anthropic_client()
    except RuntimeError:
        return "_Executive summary unavailable (ANTHROPIC_API_KEY not set)._"
    del client  # actual call deferred to prompt-engineering pass
    return "_Executive summary placeholder — wire up the prompt before publishing._"


def generate_qualitative_observations(records: list[ExecutionRecord]) -> str:
    """Produce qualitative observations across the rodada.

    TODO: prompt engineering. Sample N decisions+justifications across
    difficulty buckets and ask the model for patterns.
    """
    # TODO(prompt-engineering): replace stub with real prompt+model call.
    try:
        _anthropic_client()
    except RuntimeError:
        return "_Qualitative observations unavailable (ANTHROPIC_API_KEY not set)._"
    del records
    return "_Qualitative observations placeholder — wire up the prompt before publishing._"


def summarize_notable_cases(top_failures: list[ExecutionRecord], top_successes: list[ExecutionRecord]) -> str:
    """Write short narratives for the most informative failures and successes.

    TODO: prompt engineering. Use the records' justifications and harness
    labels as evidence; do not invent facts.
    """
    # TODO(prompt-engineering): replace stub with real prompt+model call.
    try:
        _anthropic_client()
    except RuntimeError:
        return "_Notable-cases narrative unavailable (ANTHROPIC_API_KEY not set)._"
    del top_failures, top_successes
    return "_Notable-cases placeholder — wire up the prompt before publishing._"


# --- Notable case selection ----------------------------------------------

def _select_notable(records: list[ExecutionRecord]) -> tuple[list[ExecutionRecord], list[ExecutionRecord]]:
    """Pick illustrative failure and success cases.

    Failures: FP and FN where the alternative label is known.
    Successes: TP and TN, prioritising the ones with highest token cost
    (cheapest signal that the system put real effort in).
    """
    failures = [r for r in records if r.confusion_matrix_cell in ("FP", "FN")]
    successes = [r for r in records if r.confusion_matrix_cell in ("TP", "TN")]
    failures.sort(key=lambda r: r.total_tokens, reverse=True)
    successes.sort(key=lambda r: r.total_tokens, reverse=True)
    return failures[:5], successes[:5]


# --- Report rendering ----------------------------------------------------

def _render_report_md(
    rodada_id: str,
    metrics: dict[str, Any],
    records: list[ExecutionRecord],
    out_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    error_summary: dict[str, Any] | None = None,
    incomplete_pairs: list[dict[str, Any]] | None = None,
    only_complete_pairs: bool = False,
) -> None:
    """Write the consolidated Markdown report to disk."""
    cls = metrics["classification"]
    rem = metrics["remediation"]
    cost = metrics["cost"]
    var = metrics["variability"]
    by_diff = metrics["by_difficulty"]
    by_source = metrics["by_source"]
    mutation_detection = metrics["mutation_detection"]
    counts = metrics["confusion"]
    manifest = manifest or {}
    error_summary = error_summary or {"total": 0, "by_reason": {}, "samples": []}
    incomplete_pairs = incomplete_pairs or []

    failures, successes = _select_notable(records)
    cost_estimate = estimate_cost(cost["tokens_by_agent_total"])
    communicator_status = _infer_communicator_status(records)

    lines: list[str] = []
    lines.append(f"# Rodada `{rodada_id}` — consolidated report")
    lines.append("")

    # ---- Experimental Configuration ------------------------------------
    lines.append("## Experimental configuration")
    lines.append("")
    planned_count = len(manifest.get("planned_instances", [])) if manifest else None
    reserve_count = len(manifest.get("reserve_instances", [])) if manifest else None
    effective_records = metrics["n_records"]
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    if planned_count is not None:
        lines.append(f"| Planned instances (post `--max-cases`) | {planned_count} |")
    if reserve_count is not None:
        lines.append(f"| Reserve instances (overflow) | {reserve_count} |")
    lines.append(f"| Records analysed (valid) | {effective_records} |")
    lines.append(f"| Records excluded due to errors | {error_summary['total']} |")
    if only_complete_pairs:
        lines.append("| Filter | `--only-complete-pairs` applied |")
    if manifest.get("aborted"):
        lines.append(
            f"| **Rodada aborted early** | yes — {manifest.get('abort_reason')!r} |"
        )
    lines.append(f"| Communicator status | {communicator_status} |")
    lines.append(
        f"| Total estimated cost (USD) | "
        f"${cost_estimate['total_usd_estimate']:.2f} "
        f"(assumed input share = {cost_estimate['assumed_input_share']:.0%}) |"
    )
    lines.append("")
    if manifest.get("config"):
        cfg = manifest["config"]
        lines.append("### Rodada config (from `manifest.json`)")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for key in (
            "repetitions",
            "patches_per_case",
            "patch_sources",
            "case_schedule",
            "max_cases",
            "dataset",
        ):
            if key in cfg:
                lines.append(f"| `{key}` | `{cfg[key]}` |")
        lines.append("")

    lines.append("### Expected agent → model assignment")
    lines.append("")
    lines.append(
        "Values below are the documented configuration for this experiment. "
        "MAS-Ops does not echo back the model id per agent, so this mapping "
        "is **not** verified at runtime."
    )
    lines.append("")
    lines.append("| Agent | Expected model | Tokens (sum) | Estimated USD |")
    lines.append("|---|---|---|---|")
    per_agent = cost_estimate["per_agent"]
    seen_agents = set(per_agent.keys()) | set(EXPECTED_AGENT_MODELS.keys())
    for agent in sorted(seen_agents):
        details = per_agent.get(agent) or {
            "model": EXPECTED_AGENT_MODELS.get(agent, "unknown"),
            "tokens": 0,
            "usd_estimate": 0.0,
        }
        usd = details["usd_estimate"]
        usd_str = f"${usd:.2f}" if usd is not None else "n/a"
        lines.append(
            f"| `{agent}` | `{details['model']}` | "
            f"{details['tokens']} | {usd_str} |"
        )
    lines.append("")
    if error_summary["total"]:
        lines.append("### Excluded error records")
        lines.append("")
        lines.append(f"**{error_summary['total']}** record(s) excluded from metrics.")
        lines.append("")
        if error_summary["by_reason"]:
            lines.append("| Reason fingerprint | Count |")
            lines.append("|---|---|")
            for reason, count in error_summary["by_reason"].items():
                lines.append(f"| `{reason}` | {count} |")
            lines.append("")
    if incomplete_pairs:
        lines.append("### Incomplete (instance, repetition) pairs")
        lines.append("")
        lines.append(
            f"**{len(incomplete_pairs)}** pair(s) missing at least one expected "
            "source class. Affects PRRR/PFNRR comparability across sources. "
            "Use `--only-complete-pairs` to filter these out for stricter analysis."
        )
        lines.append("")
        lines.append("| Instance | Rep | Present | Missing |")
        lines.append("|---|---|---|---|")
        for p in incomplete_pairs[:20]:
            lines.append(
                f"| `{p['instance_id']}` | {p['repetition']} | "
                f"{', '.join(p['present'])} | {', '.join(p['missing'])} |"
            )
        if len(incomplete_pairs) > 20:
            lines.append(f"| _…{len(incomplete_pairs) - 20} more rows omitted_ | | | |")
        lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append(generate_executive_summary(metrics))
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    lines.append(f"- Records analysed: **{metrics['n_records']}** "
                 f"(of which **{metrics['n_errors']}** error rows)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {cls['accuracy']:.3f} |")
    lines.append(f"| Precision | {cls['precision']:.3f} |")
    lines.append(f"| Recall | {cls['recall']:.3f} |")
    lines.append(f"| F1 | {cls['f1']:.3f} |")
    lines.append(f"| Support (TP+TN+FP+FN) | {cls['support']} |")
    lines.append("")
    lines.append("### Confusion matrix")
    lines.append("")
    lines.append("|  | Pred approve | Pred reject |")
    lines.append("|---|---|---|")
    lines.append(f"| **GT resolved** | {counts.get('TP',0)} (TP) | {counts.get('FN',0)} (FN) |")
    lines.append(f"| **GT not_resolved** | {counts.get('FP',0)} (FP) | {counts.get('TN',0)} (TN) |")
    lines.append("")
    lines.append("### Remediation rates")
    lines.append("")
    lines.append(
        f"- **PRRR** (Post-Rejection Remediation Rate): "
        f"{rem['prrr']:.3f} over {int(rem['prrr_support'])} TN cases with an alternative_patch."
    )
    lines.append(
        f"- **PFNRR** (Post-False-Negative Recovery Rate): "
        f"{rem['pfnrr']:.3f} over {int(rem['pfnrr_support'])} FN cases with an alternative_patch."
    )
    lines.append("")
    lines.append("### Cost & runtime")
    lines.append("")
    lines.append(f"- Mean total tokens per request: **{cost['total_tokens_mean']:.0f}**")
    lines.append(f"- Total tokens across rodada: **{cost['total_tokens_sum']}**")
    lines.append(f"- Mean duration (s): **{cost['duration_seconds_mean']:.1f}**")
    lines.append(f"- Mean guardian iterations: **{cost['guardian_iterations_mean']:.2f}**")
    if cost["tokens_by_agent_total"]:
        lines.append("- Tokens by agent (sum across rodada):")
        for agent, value in sorted(cost["tokens_by_agent_total"].items()):
            lines.append(f"  - `{agent}`: {value}")
    lines.append("")
    lines.append("## Stratification by difficulty")
    lines.append("")
    lines.append("| Bucket | N | Accuracy | F1 | PRRR | PFNRR | Mean tokens |")
    lines.append("|---|---|---|---|---|---|---|")
    for bucket, body in by_diff.items():
        c = body["classification"]
        r = body["remediation"]
        co = body["cost"]
        lines.append(
            f"| {bucket} | {body['n']} | {c['accuracy']:.3f} | {c['f1']:.3f} | "
            f"{r['prrr']:.3f} | {r['pfnrr']:.3f} | {co['total_tokens_mean']:.0f} |"
        )
    lines.append("")
    lines.append("## Performance by patch source")
    lines.append("")
    if by_source:
        lines.append("| Source | N | Accuracy | Precision | Recall | F1 | PRRR | PFNRR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for source, body in by_source.items():
            c = body["classification"]
            r = body["remediation"]
            lines.append(
                f"| `{source}` | {body['n']} | {c['accuracy']:.3f} | "
                f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
                f"{r['prrr']:.3f} | {r['pfnrr']:.3f} |"
            )
    else:
        lines.append("_No records._")
    lines.append("")
    lines.append("### Detection by mutation type")
    lines.append("")
    if mutation_detection:
        lines.append(
            "For each synthetic mutation, **detection rate** is the share of "
            "ground-truth `not_resolved` cases that MAS-Ops correctly rejected. "
            "Mutations whose harness label remained `resolved` are tracked "
            "separately — they exercise scope/style discipline, not correctness."
        )
        lines.append("")
        lines.append(
            "| Mutation source | N | GT not_resolved | GT resolved | "
            "Detection rate | Approve-rate when still resolved |"
        )
        lines.append("|---|---|---|---|---|---|")
        for source, body in mutation_detection.items():
            lines.append(
                f"| `{source}` | {body['n_total']} | "
                f"{body['n_ground_truth_not_resolved']} | "
                f"{body['n_ground_truth_resolved']} | "
                f"{body['detection_rate']:.3f} | "
                f"{body['approve_rate_when_still_resolved']:.3f} |"
            )
    else:
        lines.append("_No synthetic records in this rodada._")
    lines.append("")
    lines.append("## Variability across repetitions")
    lines.append("")
    lines.append(
        f"- Instances with ≥2 repetitions: **{int(var['instances_with_repetitions'])}**"
    )
    lines.append(
        f"- Instances where repetitions disagreed on the decision: "
        f"**{int(var['instances_with_decision_disagreement'])}** "
        f"({var['decision_disagreement_rate']:.1%})"
    )
    lines.append(
        f"- Mean token stdev within instance: **{var['mean_token_stdev_within_instance']:.0f}**"
    )
    lines.append("")
    lines.append("## Qualitative observations")
    lines.append("")
    lines.append(generate_qualitative_observations(records))
    lines.append("")
    lines.append("## Notable cases")
    lines.append("")
    lines.append(summarize_notable_cases(failures, successes))
    lines.append("")
    lines.append("### Top failures")
    lines.append("")
    for r in failures:
        lines.append(
            f"- `{r.instance_id}` rep={r.repetition} "
            f"({r.confusion_matrix_cell}, decided_by={r.decided_by}): "
            f"`execution_id={r.execution_id}`"
        )
    if not failures:
        lines.append("_None._")
    lines.append("")
    lines.append("### Top successes")
    lines.append("")
    for r in successes:
        lines.append(
            f"- `{r.instance_id}` rep={r.repetition} "
            f"({r.confusion_matrix_cell}, decided_by={r.decided_by}): "
            f"`execution_id={r.execution_id}`"
        )
    if not successes:
        lines.append("_None._")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append("- `charts/confusion_matrix.png`")
    lines.append("- `charts/tokens_by_difficulty.png`")
    lines.append("- `charts/accuracy_by_difficulty.png`")
    lines.append("- `charts/guardian_iterations.png`")
    lines.append("")
    lines.append("## Known limitations")
    lines.append("")
    lines.append(
        "- `tokens_by_agent` only covers agents using `llm.client.call_llm` "
        "(detective, fixer, fixer_guardian, executor_guardian). Executor and Communicator "
        "use the Claude Agent SDK and `langchain.create_agent` respectively, and their token "
        "usage must be retrieved from Langfuse traces on EC2-mas-ops."
    )
    lines.append(
        "- Execution is strictly sequential due to a known concurrency limitation in MAS-Ops. "
        "Rodada wall-clock time scales linearly with N × repetitions."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --- Orchestration ------------------------------------------------------

def aggregate(rodada_id: str, *, only_complete_pairs: bool = False) -> Path:
    """Aggregate a rodada and write report artifacts; return the output dir."""
    settings = get_settings()
    results_dir = settings.results_dir / rodada_id
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory for rodada {rodada_id!r} at {results_dir}")

    out_dir = settings.consolidated_dir / rodada_id
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    raw_records = _load_records(results_dir)
    if not raw_records:
        raise RuntimeError(f"No execution records found in {results_dir}")

    valid_records, errored_records = _split_records(raw_records)
    error_summary = _summarise_errors(errored_records)

    # Resolve the expected source classes from the manifest, defaulting to
    # the framework's standard pairing of gold + synthetic when absent.
    manifest = _load_manifest(results_dir)
    cfg = manifest.get("config", {}) if manifest else {}
    expected_classes_raw = cfg.get("patch_sources") or ["gold", "synthetic"]
    expected_classes = set(expected_classes_raw)

    incomplete_pairs = _detect_incomplete_pairs(valid_records, expected_classes)

    records: list[ExecutionRecord]
    if only_complete_pairs:
        records = _filter_complete_pairs(valid_records, expected_classes)
        dropped = len(valid_records) - len(records)
        console.log(
            f"[aggregate] --only-complete-pairs dropped {dropped} record(s) "
            f"({len(incomplete_pairs)} pair(s) incomplete)"
        )
    else:
        records = valid_records

    console.log(
        f"[aggregate] loaded {len(raw_records)} records from {results_dir} "
        f"(valid={len(valid_records)}, errored={len(errored_records)}, "
        f"analysed={len(records)})"
    )

    metrics = compute_metrics(records)

    # CSV — exports every record (valid + errored) so the trail is complete.
    df = _records_to_dataframe(raw_records)
    # Ensure candidate_patch_source is present and visible early in column order.
    if "candidate_patch_source" in df.columns:
        ordered = ["candidate_patch_source"] + [
            c for c in df.columns if c != "candidate_patch_source"
        ]
        df = df[ordered]
    df.to_csv(out_dir / "all_executions.csv", index=False)

    # Charts (operate on analysed records only — errored rows would skew).
    _render_charts(records, metrics, charts_dir)

    # Markdown report
    _render_report_md(
        rodada_id,
        metrics,
        records,
        out_dir / "report.md",
        manifest=manifest,
        error_summary=error_summary,
        incomplete_pairs=incomplete_pairs,
        only_complete_pairs=only_complete_pairs,
    )

    # Pretty summary in terminal
    table = Table(title=f"Rodada {rodada_id} — aggregated")
    cls = metrics["classification"]
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Accuracy", f"{cls['accuracy']:.3f}")
    table.add_row("Precision", f"{cls['precision']:.3f}")
    table.add_row("Recall", f"{cls['recall']:.3f}")
    table.add_row("F1", f"{cls['f1']:.3f}")
    table.add_row("PRRR", f"{metrics['remediation']['prrr']:.3f}")
    table.add_row("PFNRR", f"{metrics['remediation']['pfnrr']:.3f}")
    table.add_row("Records", str(metrics["n_records"]))
    console.print(table)
    console.log(f"[aggregate] wrote outputs to {out_dir}")
    return out_dir


# --- CLI ---------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a MAS-Ops evaluation rodada into a consolidated report.",
    )
    parser.add_argument("--rodada-id", required=True, help="Rodada identifier to aggregate.")
    parser.add_argument(
        "--only-complete-pairs",
        action="store_true",
        help=(
            "Restrict the analysis to (instance, repetition) pairs that cover "
            "every expected source class (e.g. both gold and synthetic). "
            "Useful for strict per-source comparability."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    try:
        aggregate(args.rodada_id, only_complete_pairs=args.only_complete_pairs)
    except Exception:
        logger.exception("Aggregation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
