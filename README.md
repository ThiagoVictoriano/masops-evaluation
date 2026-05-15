# masops-evaluation

Evaluation framework for the **MAS-Ops** multi-agent PR-review system against
[SWE-bench Verified](https://www.swebench.com/).

> **This repository is the experiment, not the system under test.** The
> MAS-Ops system itself (the Detective / Fixer / Guardian / Executor /
> Communicator agents) lives in a separate codebase. The scripts here drive
> SWE-bench instances through MAS-Ops in *evaluation mode* and score the
> resulting decisions against the SWE-bench ground truth.

---

## What this repo does

For every selected SWE-bench Verified instance, the framework:

1. Loads the instance metadata (issue, base commit, gold patch, hidden tests,
   difficulty).
2. Picks a **candidate patch** (the gold patch for now; trajectory and
   synthetic mutations are planned).
3. Runs the SWE-bench harness on that patch to obtain a **ground-truth
   label** (`resolved` / `not_resolved`).
4. POSTs a synthetic PR review request to the MAS-Ops `/eval/pr-review`
   endpoint.
5. Captures the MAS-Ops decision (`approve` / `reject`), its justification,
   and any `alternative_patch` it proposed.
6. If an `alternative_patch` was returned, runs the harness again to label
   that too.
7. Writes a structured `ExecutionRecord` JSON per (instance, repetition).

A separate aggregation step turns those JSONs into a confusion matrix,
classification metrics (accuracy/precision/recall/F1), remediation rates
(PRRR, PFNRR), per-difficulty stratifications, repetition variability, and
a Markdown report with charts.

---

## Architecture

```
+----------------------------+                +----------------------------+
|  EC2-benchmark             |  HTTP (VPC)    |  EC2-mas-ops               |
|                            |  POST /eval/   |                            |
|  - this repo               | ─ pr-review ─► |  MAS-Ops in eval mode      |
|  - SWE-bench harness       |                |  - Detective               |
|  - Docker (for harness)    | ◄── 200 JSON ─ |  - Fixer / Guardian loop   |
|                            |                |  - Executor / Communicator |
+----------------------------+                +----------------------------+
```

- Communication is **synchronous HTTP** over the private VPC IP.
- Execution is strictly **sequential**: MAS-Ops is known to mishandle
  overlapping `/eval/pr-review` requests, so the orchestrator never overlaps
  them. Wall-clock time scales linearly with `N × repetitions`.

---

## Request / response contract

**Request** — `POST http://{MASOPS_HOST}:{MASOPS_PORT}/eval/pr-review`

```json
{
  "_eval_metadata": {
    "instance_id": "django__django-11848",
    "candidate_patch": "diff --git a/...",
    "base_commit": "f4e93919...",
    "repo": "django/django"
  },
  "pull_request": {
    "number": 1,
    "title": "Eval: django__django-11848",
    "body": "<problem_statement of the SWE-bench issue>"
  },
  "repository": {
    "full_name": "django/django"
  }
}
```

**Response**

```json
{
  "decision": "approve" | "reject",
  "decided_by": "detective" | "fixer_guardian_loop",
  "justification": "...",
  "alternative_patch": "<diff>" | null,
  "alternative_metadata": {
    "guardian_iterations": 0,
    "guardian_final_verdict": "..." | null
  },
  "execution_metadata": {
    "total_tokens": 12345,
    "tokens_by_agent": { "detective": 1000, "fixer": 234 },
    "agents_invoked": ["detective"],
    "duration_seconds": 12.5
  }
}
```

`tokens_by_agent` only covers agents that go through `llm.client.call_llm`
(`detective`, `fixer`, `fixer_guardian`, `executor_guardian`). Executor and
Communicator use the Claude Agent SDK and `langchain.create_agent`
respectively; their costs are visible via Langfuse traces on EC2-mas-ops
rather than this field. When MAS-Ops cannot report any per-agent breakdown,
the field is the empty dict `{}` (never `null`).

**HTTP statuses we handle**

| Status | Meaning                                                                                  | Client behavior            |
|--------|------------------------------------------------------------------------------------------|----------------------------|
| `200`  | Normal response (includes rejections with `alternative_patch: null`).                    | Parse and persist.         |
| `400`  | Malformed payload (missing field, `_eval_metadata` absent, etc.).                        | No retry; log and persist. |
| `404`  | Endpoint unavailable (MAS-Ops not running in evaluation mode).                           | No retry; abort.           |
| `5xx`  | Internal MAS-Ops error.                                                                  | Retry with exponential backoff (2s / 4s / 8s). |

---

## Prerequisites

- **EC2 instance**: at least **16 GB RAM**, **200 GB disk**. The SWE-bench
  harness clones large repositories (Django, sympy, sklearn, etc.) on demand
  and runs each instance inside a fresh Docker container.
- **Docker** (required by the SWE-bench harness).
- **Python 3.11+**.
- **Network reachability** to the MAS-Ops EC2 on its private IP and configured
  port.
- **Anthropic API key** — *optional*; only needed if you want the
  aggregator to produce LLM-generated narrative sections (executive
  summary, qualitative observations, notable cases). Skip the key and
  the `[narrative]` extra and those sections render as placeholders;
  every quantitative metric still computes normally.

---

## Setup

On a fresh EC2-benchmark instance, run the bootstrap script:

```bash
git clone <this-repo-url> masops-evaluation
cd masops-evaluation
bash scripts/setup-ec2.sh
```

The script installs Docker, Python 3.11+, creates a virtualenv at `.venv`,
and installs this package in editable mode with dev extras.

For local development:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# Optional: add the [narrative] extra to enable Anthropic-backed
# narrative sections in the aggregator. Skip if you do not have an
# Anthropic API key — the rest of the framework works without it.
# pip install -e ".[dev,narrative]"
```

---

## Configuration

Copy the template and fill in the values:

```bash
cp .env.example .env
```

Key variables (see `.env.example` for the full list):

| Variable                          | Purpose                                                          |
|-----------------------------------|------------------------------------------------------------------|
| `MASOPS_HOST`                     | Private VPC IP of EC2-mas-ops.                                   |
| `MASOPS_PORT`                     | MAS-Ops API port (default `8000`).                               |
| `MASOPS_TIMEOUT`                  | Per-request HTTP timeout in seconds (default `1200`).            |
| `MASOPS_HEALTH_PATH`              | Path of the MAS-Ops health endpoint (default `/health`).         |
| `SWEBENCH_DATASET`                | Hugging Face dataset id (default `princeton-nlp/SWE-bench_Verified`). |
| `SWEBENCH_HARNESS_RUN_ID_PREFIX`  | Prefix used to build harness `run_id`s (default `eval`).         |
| `ANTHROPIC_API_KEY`               | *Optional.* Enables LLM narrative sections; leave empty to skip them. |
| `RESULTS_DIR`                     | Where per-execution JSONs land (default `./results`).            |
| `CONSOLIDATED_DIR`                | Where aggregator outputs land (default `./consolidated`).        |
| `LOG_LEVEL`                       | `DEBUG` / `INFO` / `WARNING` / `ERROR`.                          |

---

## Running an evaluation

Three commands, in order:

```bash
# 1. Select instances stratified by difficulty (one-off per study).
select-instances --n-per-difficulty 15 --seed 42

# 2. Run the rodada (sequential; resumable).
run-evaluation --rodada-id v1 --repetitions 3

# 3. Aggregate into a Markdown report with charts and a CSV.
aggregate-results --rodada-id v1
```

Every command can also be invoked via `python -m masops_evaluation.<module>`,
which is handy when the entry points haven't been re-installed after a code
change.

### Useful flags

- `run-evaluation --resume` skips `(instance, repetition, source)` triples
  that already have a result file under `results/{rodada_id}/`.
- `run-evaluation --instance-ids django__django-11848 ...` runs a custom
  subset (handy for pilots and debugging) — overrides the selection file.
- `run-evaluation --patch-sources gold,synthetic --patches-per-case 2`
  (the defaults) evaluates one gold patch and one synthetic mutation per
  `(instance, repetition)`. Pass `--patch-sources gold` to evaluate gold
  only; pass `--patches-per-case 4` to expand the per-case fan-out (the
  schedule round-robins the listed sources).

---

## Patch sources

The framework supports two patch source classes, selected via
`--patch-sources`:

### `gold`

The pristine SWE-bench gold patch — the human-authored fix that ships with
the dataset. By construction the harness labels these as `resolved`, so
they probe **True Positive** and **False Negative** behaviour: does
MAS-Ops correctly approve a good fix, or does it spuriously reject it?

### `synthetic`

A deterministic mutation of the gold patch produced by
[`mutations.py`](src/masops_evaluation/mutations.py). Each mutation
exercises a specific defect class that the Guardian should catch:

| Mutation (`MutationType`) | What it changes                                                                  | Defect class probed                       |
|---------------------------|----------------------------------------------------------------------------------|-------------------------------------------|
| `remove_addition`         | Removes one random `+` line.                                                     | Incomplete fix.                           |
| `invert_conditional`      | Flips one `==/!=/</>/<=/>=/and/or/is/is not` in an added line.                   | Inverted logic.                           |
| `expand_scope`            | Appends an unrelated new-file modification on top of the gold patch.             | Scope discipline.                         |
| `remove_critical_line`    | Removes the most load-bearing `+` line (heuristic: `return`/`raise`/`assert`, else longest). | Effect-less fix. |

When `synthetic` is selected, the mutation type cycles across the rodada
to balance attempts across the four classes. The concrete type chosen for
each execution is recorded in `candidate_patch_source` as one of
`synthetic_remove_addition`, `synthetic_invert_conditional`,
`synthetic_expand_scope`, or `synthetic_remove_critical_line`.

Each mutation is deterministic given `(gold_patch, seed)`, where the seed
is derived from `(rodada_id, instance_id, repetition, mutation_type)` —
rerunning the rodada with the same coordinates reproduces the same
mutated diffs.

**Methodological note on `expand_scope`.** This mutation preserves the
original fix and adds an unrelated change, so the harness may still label
the result as `resolved`. That is expected: `expand_scope` is a scope-
discipline test rather than a clean negative case, and the aggregator's
"Detection by mutation type" table tracks both outcomes separately.

### `trajectory` (future work)

Real failed-agent trajectories from the public
[`SWE-bench/experiments`](https://github.com/SWE-bench/experiments) corpus
and similar sources (SWE-agent / Aider / OpenHands submissions) would
yield more authentic negative examples than synthetic mutations. That
data pipeline is **not implemented yet** — it is left as future work
because it requires curating, normalising, and licensing external
submission corpora.

---

## Cost considerations

The reference rodada is sized to fit inside a **~$20 USD** API budget,
which constrains how many cases and how many MAS-Ops sub-agents we can
afford to invoke per rodada.

### Defaults aligned with the budget

- **10 cases × 2 sources × 1 repetition = 20 executions per rodada.**
  Defaults: `select-instances --n-per-difficulty 4` (12 candidates,
  10 used by `run-evaluation` plus 2 reserve); `run-evaluation
  --max-cases 10 --repetitions 1 --patches-per-case 2`.
- **Communicator is expected to be disabled** in eval mode on the
  MAS-Ops side via `EVAL_INVOKE_COMMUNICATOR=false` in EC2-mas-ops'
  `.env`. Communicator output adds tokens without changing the
  decision, so disabling it materially reduces cost per request.
- **Estimated rodada cost: $7–$14 USD** with the documented model
  assignment (Gemini 3.1 Pro Preview for Detective/Fixer, Claude Sonnet
  4.6 for both Guardian loops and the Executor). The aggregator's
  *Experimental configuration* section prints the post-hoc estimate
  using `MODEL_PRICING_USD_PER_MTOK` in
  [`aggregate_results.py`](src/masops_evaluation/aggregate_results.py).
- Margin for **one or two additional rodadas** (e.g. another 10–15
  cases, or one repetition more) is intentional — leave headroom in case
  a debug rodada is needed.

### Abort-on-budget behaviour

`run-evaluation` watches for budget signals in MAS-Ops error responses:
HTTP `402`/`429`, or any error message containing `billing`, `credit`,
or `quota` (case-insensitive). When any of those is detected, the
rodada writes the failed record, marks `aborted=true` and
`abort_reason` in `manifest.json`, and exits non-zero. All other
errors (timeouts, 5xx, network issues) only emit an `error_message`
on the record and let the rodada continue.

### Recommended provider-side alerts

Configure a hard cap and threshold alerts at your LLM provider
(OpenRouter / Anthropic / Google) at **80%** and **95%** of the rodada
budget. These run independently from this framework's abort logic and
protect against MAS-Ops-side bugs that fail to surface upstream errors.

### Pricing assumptions

The cost estimate divides per-agent token totals 50/50 between input
and output (override via `assumed_input_share` if your workload skews
differently). Prices for known models live in
`MODEL_PRICING_USD_PER_MTOK` and need to be updated as upstream
pricing changes. Agents whose tokens land outside that dict (or whose
model assignment is unknown) are reported with an `n/a` cost cell.

---

## Repository layout

```
masops-evaluation/
├── .env.example                      # env-var template
├── pyproject.toml                    # PEP 621 packaging + tool config
├── README.md                         # this file
├── scripts/
│   └── setup-ec2.sh                  # EC2-benchmark bootstrap
├── data/
│   └── selected_instances.json      # written by select-instances
├── results/                          # per-execution JSONs (gitignored)
│   └── {rodada_id}/
│       ├── manifest.json
│       └── {instance_id}__rep-{n}__src-{source}.json
├── consolidated/                     # aggregator outputs (gitignored)
│   └── {rodada_id}/
│       ├── report.md
│       ├── all_executions.csv
│       └── charts/
├── src/
│   └── masops_evaluation/
│       ├── __init__.py
│       ├── config.py                 # Settings (pydantic-settings)
│       ├── schemas.py                # Pydantic models + ExecutionRecord
│       ├── harness_client.py         # SWE-bench harness wrapper
│       ├── masops_client.py          # HTTP client for MAS-Ops
│       ├── mutations.py              # Deterministic synthetic patch mutations
│       ├── select_instances.py       # CLI: select-instances
│       ├── run_evaluation.py         # CLI: run-evaluation
│       └── aggregate_results.py      # CLI: aggregate-results
└── tests/
    ├── test_schemas.py               # Pydantic round-trip tests
    └── test_mutations.py             # Mutation determinism + diff validity tests
```

---

## Known limitations

- **Sequential execution is mandatory.** MAS-Ops cannot safely handle
  concurrent `/eval/pr-review` requests, so the orchestrator processes one
  `(instance, repetition)` at a time. Plan rodada sizes accordingly.
- **Partial cost coverage.** `tokens_by_agent` excludes Executor (Claude
  Agent SDK) and Communicator (langchain). For end-to-end token accounting,
  consult Langfuse traces on EC2-mas-ops.
- **Patch sources beyond `gold` and `synthetic`.** Real failed-agent
  `trajectory` data is not wired up yet; see the "Patch sources" section
  for the motivation.
- **Aggregator narrative is a stub.** `generate_executive_summary`,
  `generate_qualitative_observations`, and `summarize_notable_cases` are
  wired but contain `TODO` prompts — they emit placeholders until that
  prompt engineering lands.

---

## References

- SWE-bench: <https://www.swebench.com/>
- SWE-bench Verified dataset: <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified>
- MAS-Ops: the multi-agent PR-review system under evaluation (separate
  repository, internal to the project).
