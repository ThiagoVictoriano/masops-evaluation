"""CLI: select N SWE-bench Verified instances stratified by difficulty.

The SWE-bench Verified dataset ships with a free-text ``difficulty`` field.
We collapse the upstream labels into three buckets — ``easy``, ``medium``,
``hard`` — and sample ``--n-per-difficulty`` ids from each bucket with a
fixed seed so the selection is reproducible across runs.

The selection is written to JSON for downstream consumption by
``run-evaluation``. The default ``--n-per-difficulty`` is sized for the
budget-constrained reference rodada (4 per bucket × 3 buckets = 12
instances; ``run-evaluation`` consumes the first 10 by default via
``--max-cases``, leaving 2 as overflow reserve for manual replacement
when a case fails the pipeline end to end).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from masops_evaluation.config import configure_logging, get_settings

logger = logging.getLogger(__name__)
console = Console()


# Upstream label -> bucket mapping. Keep keys lowercase for comparisons.
DIFFICULTY_BUCKETS: dict[str, str] = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "hard",
}

BUCKETS = ("easy", "medium", "hard")


def _bucket_for(raw: Any) -> str | None:
    """Map a raw upstream difficulty label to one of our three buckets."""
    if not isinstance(raw, str):
        return None
    return DIFFICULTY_BUCKETS.get(raw.strip().lower()) or DIFFICULTY_BUCKETS.get(raw.strip())


def _load_dataset_rows(dataset_name: str) -> list[dict[str, Any]]:
    """Load the dataset and return every row as a plain dict.

    Args:
        dataset_name: Hugging Face dataset identifier.

    Returns:
        A list of row dictionaries (one per instance).
    """
    from datasets import load_dataset

    console.log(f"[select] loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    rows = [dict(row) for row in dataset]
    console.log(f"[select] loaded {len(rows)} instances")
    return rows


def _group_by_bucket(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group instance ids by collapsed difficulty bucket."""
    grouped: dict[str, list[str]] = {b: [] for b in BUCKETS}
    skipped = 0
    for row in rows:
        bucket = _bucket_for(row.get("difficulty"))
        if bucket is None:
            skipped += 1
            continue
        grouped[bucket].append(row["instance_id"])
    if skipped:
        logger.warning("Skipped %d rows with unknown difficulty", skipped)
    return grouped


def _sample(grouped: dict[str, list[str]], n: int, seed: int) -> dict[str, list[str]]:
    """Sample ``n`` ids from each bucket with the given seed."""
    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for bucket in BUCKETS:
        pool = sorted(grouped[bucket])  # sort for determinism
        if len(pool) < n:
            logger.warning(
                "Bucket %s has only %d instances (<%d requested); taking all",
                bucket,
                len(pool),
                n,
            )
            sampled[bucket] = pool
        else:
            sampled[bucket] = rng.sample(pool, n)
        sampled[bucket].sort()
    return sampled


def _render_table(sampled: dict[str, list[str]], n_per_bucket: int) -> None:
    """Print a pretty summary of the selection."""
    table = Table(title="Selected instances by difficulty bucket")
    table.add_column("Bucket", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Sample IDs", style="dim")
    for bucket in BUCKETS:
        ids = sampled[bucket]
        sample = ", ".join(ids[:3]) + ("…" if len(ids) > 3 else "")
        table.add_row(bucket, f"{len(ids)}/{n_per_bucket}", sample)
    console.print(table)


def select_instances(
    n_per_difficulty: int,
    seed: int,
    output_path: Path,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Run the selection end to end.

    Args:
        n_per_difficulty: How many instances to sample per bucket.
        seed: Random seed.
        output_path: Where to write the JSON manifest.
        dataset_name: Override for the dataset; defaults to settings.

    Returns:
        The dictionary that was written to disk.
    """
    settings = get_settings()
    dataset = dataset_name or settings.swebench_dataset

    rows = _load_dataset_rows(dataset)
    grouped = _group_by_bucket(rows)
    sampled = _sample(grouped, n_per_difficulty, seed)

    all_ids: list[str] = []
    for bucket in BUCKETS:
        all_ids.extend(sampled[bucket])

    payload = {
        "metadata": {
            "seed": seed,
            "n_per_difficulty": n_per_difficulty,
            "dataset": dataset,
            "selection_date": datetime.now(timezone.utc).isoformat(),
        },
        "by_difficulty": sampled,
        "all_ids": all_ids,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.log(f"[select] wrote {output_path}")
    _render_table(sampled, n_per_difficulty)

    # Overflow / reserve advisory — the framework's default --max-cases is 10.
    DEFAULT_MAX_CASES = 10
    total = len(all_ids)
    if total > DEFAULT_MAX_CASES:
        reserve = total - DEFAULT_MAX_CASES
        console.log(
            f"[select] selection has {total} instances; the default rodada uses the "
            f"first {DEFAULT_MAX_CASES} (via run-evaluation --max-cases). "
            f"The remaining {reserve} are reserved for manual replacement if a "
            "case fails the pipeline end-to-end."
        )
    elif total < DEFAULT_MAX_CASES:
        console.log(
            f"[select] selection has only {total} instances — below the default "
            f"--max-cases of {DEFAULT_MAX_CASES}. The full selection will be used "
            "and there is no overflow reserve."
        )
    else:
        console.log(
            f"[select] selection has exactly {total} instances; no overflow reserve."
        )
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select SWE-bench Verified instances stratified by difficulty.",
    )
    parser.add_argument(
        "--n-per-difficulty",
        type=int,
        default=4,
        help=(
            "Instances to sample per bucket (default: 4). "
            "Total selection size = N * 3 buckets. With the default, "
            "run-evaluation will use the first 10 instances and keep 2 "
            "as an overflow reserve for manual replacement."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selected_instances.json"),
        help="Output JSON path (default: data/selected_instances.json).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the Hugging Face dataset (defaults to env SWEBENCH_DATASET).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point used by both the CLI script and ``python -m`` invocation."""
    args = _parse_args(argv)
    configure_logging()
    try:
        select_instances(
            n_per_difficulty=args.n_per_difficulty,
            seed=args.seed,
            output_path=args.output,
            dataset_name=args.dataset,
        )
    except Exception:
        logger.exception("Instance selection failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
