# Rodada `rodada-1` — consolidated report

## Experimental configuration

| Item | Value |
|---|---|
| Planned instances (post `--max-cases`) | 30 |
| Reserve instances (overflow) | 6 |
| Records analysed (valid) | 60 |
| Records excluded due to errors | 0 |
| Communicator status | active (observed in agents_invoked) |
| Total estimated cost (USD) | $2.02 (assumed input share = 50%) |

### Rodada config (from `manifest.json`)

| Key | Value |
|---|---|
| `repetitions` | `1` |
| `patches_per_case` | `2` |
| `patch_sources` | `['gold', 'synthetic']` |
| `case_schedule` | `['gold', 'synthetic']` |
| `max_cases` | `30` |
| `dataset` | `princeton-nlp/SWE-bench_Verified` |

### Expected agent → model assignment

Values below are the documented configuration for this experiment. MAS-Ops does not echo back the model id per agent, so this mapping is **not** verified at runtime.

| Agent | Expected model | Tokens (sum) | Estimated USD |
|---|---|---|---|
| `detective` | `google/gemini-3.1-pro-preview` | 164534 | $1.15 |
| `executor` | `anthropic/claude-sonnet-4.6` | 0 | $0.00 |
| `executor_guardian` | `anthropic/claude-sonnet-4.6` | 52661 | $0.47 |
| `fixer` | `google/gemini-3.1-pro-preview` | 33752 | $0.24 |
| `fixer_guardian` | `anthropic/claude-sonnet-4.6` | 17071 | $0.15 |

### Custo Estimado (instrumentação interna)

Baseado em `tokens_by_agent` dos `ExecutionRecord`s. **NOTA**: subestima o custo real porque não captura o Executor (Claude Agent SDK) nem o Communicator (`langchain.create_agent`).

| Modelo | Tokens | USD estimado |
|---|---|---|
| `anthropic/claude-sonnet-4.6` | 69732 | $0.63 |
| `google/gemini-3.1-pro-preview` | 198286 | $1.39 |
| **Total estimado** | **268018** | **$2.02** |

### Custo Real (medido via Langfuse)

_Não foi possível coletar dados do Langfuse. Confira credenciais em `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) e a acessibilidade de `LANGFUSE_URL` a partir da EC2-benchmark._

### Comparação

_Sem dados do Langfuse — comparação indisponível._

## Executive summary

_Executive summary unavailable (ANTHROPIC_API_KEY is not set)._

## Overall metrics

- Records analysed: **60** (of which **0** error rows)

| Metric | Value |
|---|---|
| Accuracy | 0.898 |
| Precision | 0.886 |
| Recall | 0.975 |
| F1 | 0.929 |
| Support (TP+TN+FP+FN) | 59 |

### Confusion matrix

|  | Pred approve | Pred reject |
|---|---|---|
| **GT resolved** | 39 (TP) | 1 (FN) |
| **GT not_resolved** | 5 (FP) | 14 (TN) |

### Remediation rates

- **PRRR** (Post-Rejection Remediation Rate): 0.750 over 4 TN cases with an alternative_patch.
- **PFNRR** (Post-False-Negative Recovery Rate): 0.000 over 0 FN cases with an alternative_patch.

### Cost & runtime

- Mean total tokens per request: **4467**
- Total tokens across rodada: **268018**
- Mean duration (s): **35.1**
- Mean guardian iterations: **0.27**
- Tokens by agent (sum across rodada):
  - `detective`: 164534
  - `executor_guardian`: 52661
  - `fixer`: 33752
  - `fixer_guardian`: 17071

## Stratification by difficulty

| Bucket | N | Accuracy | F1 | PRRR | PFNRR | Mean tokens |
|---|---|---|---|---|---|---|
| easy | 24 | 0.917 | 0.941 | 1.000 | 0.000 | 3942 |
| hard | 12 | 1.000 | 1.000 | 1.000 | 0.000 | 4961 |
| medium | 24 | 0.826 | 0.875 | 0.000 | 0.000 | 4745 |

## Performance by patch source

| Source | N | Accuracy | Precision | Recall | F1 | PRRR | PFNRR |
|---|---|---|---|---|---|---|---|
| `gold` | 30 | 0.967 | 0.967 | 1.000 | 0.983 | 0.000 | 0.000 |
| `synthetic_expand_scope` | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `synthetic_invert_conditional` | 7 | 0.857 | 0.667 | 1.000 | 0.800 | 1.000 | 0.000 |
| `synthetic_remove_addition` | 8 | 0.857 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| `synthetic_remove_critical_line` | 7 | 0.571 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Detection by mutation type

For each synthetic mutation, **detection rate** is the share of ground-truth `not_resolved` cases that MAS-Ops correctly rejected. Mutations whose harness label remained `resolved` are tracked separately — they exercise scope/style discipline, not correctness.

| Mutation source | N | GT not_resolved | GT resolved | Detection rate | Approve-rate when still resolved |
|---|---|---|---|---|---|
| `synthetic_expand_scope` | 8 | 0 | 8 | 0.000 | 1.000 |
| `synthetic_invert_conditional` | 7 | 5 | 2 | 0.800 | 1.000 |
| `synthetic_remove_addition` | 8 | 7 | 0 | 0.857 | 0.000 |
| `synthetic_remove_critical_line` | 7 | 6 | 1 | 0.667 | 0.000 |

## Variability across repetitions

- Instances with ≥2 repetitions: **30**
- Instances where repetitions disagreed on the decision: **16** (53.3%)
- Mean token stdev within instance: **2698**

## Qualitative observations

_Qualitative observations unavailable (ANTHROPIC_API_KEY is not set)._

## Notable cases

_Notable-cases narrative unavailable (ANTHROPIC_API_KEY is not set)._

### Top failures

- `django__django-16032` rep=1 (FN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-django__django-16032-rep1-synthetic_remove_critical_line-229f662e`
- `sphinx-doc__sphinx-9658` rep=1 (FP, decided_by=detective): `execution_id=rodada-1-sphinx-doc__sphinx-9658-rep1-synthetic_remove_critical_line-0529e6c0`
- `django__django-16502` rep=1 (FP, decided_by=detective): `execution_id=rodada-1-django__django-16502-rep1-synthetic_invert_conditional-3490b5ce`
- `astropy__astropy-8707` rep=1 (FP, decided_by=detective): `execution_id=rodada-1-astropy__astropy-8707-rep1-gold-f64b59bf`
- `django__django-14580` rep=1 (FP, decided_by=detective): `execution_id=rodada-1-django__django-14580-rep1-synthetic_remove_addition-7fb0590f`

### Top successes

- `astropy__astropy-8707` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-astropy__astropy-8707-rep1-synthetic_remove_addition-41427478`
- `django__django-12193` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-django__django-12193-rep1-synthetic_invert_conditional-ba8df5f9`
- `django__django-10914` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-django__django-10914-rep1-synthetic_remove_addition-735ce84c`
- `django__django-13837` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-django__django-13837-rep1-synthetic_invert_conditional-8e06cd76`
- `django__django-11087` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-1-django__django-11087-rep1-synthetic_remove_critical_line-6373a163`

## Charts

- `charts/confusion_matrix.png`
- `charts/tokens_by_difficulty.png`
- `charts/accuracy_by_difficulty.png`
- `charts/guardian_iterations.png`

## Known limitations

- `tokens_by_agent` only covers agents using `llm.client.call_llm` (detective, fixer, fixer_guardian, executor_guardian). Executor and Communicator use the Claude Agent SDK and `langchain.create_agent` respectively, and their token usage must be retrieved from Langfuse traces on EC2-mas-ops.
- Execution is strictly sequential due to a known concurrency limitation in MAS-Ops. Rodada wall-clock time scales linearly with N × repetitions.
