"""Thin wrapper over the SWE-bench harness.

The harness is a CLI: ``python -m swebench.harness.run_evaluation``. We drive
it via subprocess (rather than importing it as a library) so we stay
resilient to internal refactors of upstream and so each evaluation is
isolated in its own process.

The harness writes a JSON report to a deterministic path under the current
working directory. We parse that report to extract whether the candidate
patch resolved the instance.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from rich.console import Console

from masops_evaluation.config import get_settings
from masops_evaluation.schemas import PatchLabel

logger = logging.getLogger(__name__)
console = Console()


class HarnessError(RuntimeError):
    """Raised when the harness fails in an unrecoverable way."""


def _sanitize_run_id(value: str) -> str:
    """Strip characters the harness rejects in run identifiers."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _build_run_id(suffix: str) -> str:
    """Compose a unique run_id from the configured prefix and a suffix."""
    settings = get_settings()
    return _sanitize_run_id(f"{settings.swebench_harness_run_id_prefix}-{suffix}")


def get_harness_report_path(run_id: str, instance_id: str) -> Path:
    """Return the expected report path for a (run_id, instance_id) pair.

    The harness writes reports to
    ``./logs/run_evaluation/{run_id}/{model}/{instance_id}/report.json``,
    but the model dir name we control via predictions, so we glob for it.
    """
    base = Path("logs") / "run_evaluation" / run_id
    candidates = list(base.glob(f"*/{instance_id}/report.json"))
    if not candidates:
        return base / "<unknown>" / instance_id / "report.json"
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _write_predictions_file(
    instance_id: str,
    patch_str: str,
    model_name: str,
    target_dir: Path,
) -> Path:
    """Write a single-line predictions JSONL file consumed by the harness."""
    target_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = target_dir / "predictions.jsonl"
    payload = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch_str,
    }
    predictions_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return predictions_path

def _parse_report(report_path: Path, instance_id: str) -> PatchLabel:
    """Parse the harness report and map it to a :data:`PatchLabel`.

    The harness writes per-instance reports in the format:
        {
            "<instance_id>": {
                "resolved": true|false,
                "patch_successfully_applied": true|false,
                ...
            }
        }

    Aggregate runs use a different format with ``resolved_ids``,
    ``unresolved_ids``, and ``error_ids``. We try the per-instance format
    first, then fall back to the aggregate format.
    """
    if not report_path.exists():
        logger.warning("Harness report missing at %s", report_path)
        return "error"
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse harness report %s: %s", report_path, exc)
        return "error"

    # Per-instance format (default for single-instance runs)
    instance_data = data.get(instance_id)
    if isinstance(instance_data, dict):
        if not instance_data.get("patch_successfully_applied", True):
            logger.warning("Patch failed to apply for %s", instance_id)
            return "error"
        return "resolved" if instance_data.get("resolved", False) else "not_resolved"

    # Aggregate format fallback
    resolved_ids = set(data.get("resolved_ids", []) or [])
    unresolved_ids = set(data.get("unresolved_ids", []) or [])
    error_ids = set(data.get("error_ids", []) or [])
    if instance_id in resolved_ids:
        return "resolved"
    if instance_id in unresolved_ids:
        return "not_resolved"
    if instance_id in error_ids:
        return "error"

    logger.warning("Instance %s not present in harness report buckets", instance_id)
    return "error"


def run_harness(
    instance_id: str,
    patch_str: str,    run_id_suffix: str,
    *,
    timeout_seconds: int = 1800,
    max_workers: int = 1,
) -> tuple[Literal["resolved", "not_resolved", "error"], Path]:
    """Run the SWE-bench harness on a single (instance, patch) pair.

    Args:
        instance_id: SWE-bench instance identifier.
        patch_str: Diff to apply against the instance's base commit.
        run_id_suffix: Suffix appended to the configured run_id prefix.
        timeout_seconds: Hard subprocess timeout. Defaults to 30 minutes.
        max_workers: Harness worker count. Kept at 1 because the framework
            runs sequentially.

    Returns:
        A ``(label, report_path)`` tuple. ``label`` is one of
        ``"resolved"``, ``"not_resolved"``, ``"error"``. ``report_path``
        is the (possibly non-existent) path to the harness report file
        so the caller can persist a pointer to it.
    """
    settings = get_settings()
    run_id = _build_run_id(run_id_suffix)
    model_name = "masops-eval"

    console.log(f"[harness] instance={instance_id} run_id={run_id}")

    with tempfile.TemporaryDirectory(prefix="masops-eval-") as tmpdir:
        predictions_path = _write_predictions_file(
            instance_id=instance_id,
            patch_str=patch_str,
            model_name=model_name,
            target_dir=Path(tmpdir),
        )

        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            settings.swebench_dataset,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            instance_id,
            "--cache_level",
            "instance",
        ]

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("Harness timed out after %ss for %s", timeout_seconds, instance_id)
            return "error", get_harness_report_path(run_id, instance_id)
        except OSError as exc:
            logger.error("Harness invocation failed: %s", exc)
            return "error", get_harness_report_path(run_id, instance_id)

        if completed.returncode != 0:
            logger.warning(
                "Harness exited with code %s for %s\nstderr (tail):\n%s",
                completed.returncode,
                instance_id,
                "\n".join((completed.stderr or "").splitlines()[-20:]),
            )

    report_path = get_harness_report_path(run_id, instance_id)
    label = _parse_report(report_path, instance_id)
    console.log(f"[harness] instance={instance_id} label={label}")
    return label, report_path
