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
) -> None:
    """Write the consolidated Markdown report to disk."""
    cls = metrics["classification"]
    rem = metrics["remediation"]
    cost = metrics["cost"]
    var = metrics["variability"]
    by_diff = metrics["by_difficulty"]
    counts = metrics["confusion"]

    failures, successes = _select_notable(records)

    lines: list[str] = []
    lines.append(f"# Rodada `{rodada_id}` — consolidated report")
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

def aggregate(rodada_id: str) -> Path:
    """Aggregate a rodada and write report artifacts; return the output dir."""
    settings = get_settings()
    results_dir = settings.results_dir / rodada_id
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory for rodada {rodada_id!r} at {results_dir}")

    out_dir = settings.consolidated_dir / rodada_id
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(results_dir)
    if not records:
        raise RuntimeError(f"No execution records found in {results_dir}")

    console.log(f"[aggregate] loaded {len(records)} records from {results_dir}")

    metrics = compute_metrics(records)

    # CSV
    df = _records_to_dataframe(records)
    df.to_csv(out_dir / "all_executions.csv", index=False)

    # Charts
    _render_charts(records, metrics, charts_dir)

    # Markdown report
    _render_report_md(rodada_id, metrics, records, out_dir / "report.md")

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()
    try:
        aggregate(args.rodada_id)
    except Exception:
        logger.exception("Aggregation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
