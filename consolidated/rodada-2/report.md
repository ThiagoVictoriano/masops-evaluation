# Rodada `rodada-2` — consolidated report

## Experimental configuration

| Item | Value |
|---|---|
| Planned instances (post `--max-cases`) | 30 |
| Reserve instances (overflow) | 6 |
| Records analysed (valid) | 60 |
| Records excluded due to errors | 0 |
| Communicator status | active (observed in agents_invoked) |
| Total estimated cost (USD) | $2.80 (assumed input share = 50%) |

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
| `detective` | `google/gemini-3.1-pro-preview` | 165021 | $1.16 |
| `executor` | `anthropic/claude-sonnet-4.6` | 0 | $0.00 |
| `executor_guardian` | `anthropic/claude-sonnet-4.6` | 122115 | $1.10 |
| `fixer` | `google/gemini-3.1-pro-preview` | 47526 | $0.33 |
| `fixer_guardian` | `anthropic/claude-sonnet-4.6` | 23619 | $0.21 |

### Custo Estimado (instrumentação interna)

Baseado em `tokens_by_agent` dos `ExecutionRecord`s. **NOTA**: subestima o custo real porque não captura o Executor (Claude Agent SDK) nem o Communicator (`langchain.create_agent`).

| Modelo | Tokens | USD estimado |
|---|---|---|
| `anthropic/claude-sonnet-4.6` | 145734 | $1.31 |
| `google/gemini-3.1-pro-preview` | 212547 | $1.49 |
| **Total estimado** | **358281** | **$2.80** |

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
| Accuracy | 0.881 |
| Precision | 0.900 |
| Recall | 0.923 |
| F1 | 0.911 |
| Support (TP+TN+FP+FN) | 59 |

### Confusion matrix

|  | Pred approve | Pred reject |
|---|---|---|
| **GT resolved** | 36 (TP) | 3 (FN) |
| **GT not_resolved** | 4 (FP) | 16 (TN) |

### Remediation rates

- **PRRR** (Post-Rejection Remediation Rate): 0.833 over 6 TN cases with an alternative_patch.
- **PFNRR** (Post-False-Negative Recovery Rate): 1.000 over 1 FN cases with an alternative_patch.

### Cost & runtime

- Mean total tokens per request: **5971**
- Total tokens across rodada: **358281**
- Mean duration (s): **45.8**
- Mean guardian iterations: **0.40**
- Tokens by agent (sum across rodada):
  - `detective`: 165021
  - `executor_guardian`: 122115
  - `fixer`: 47526
  - `fixer_guardian`: 23619

## Stratification by difficulty

| Bucket | N | Accuracy | F1 | PRRR | PFNRR | Mean tokens |
|---|---|---|---|---|---|---|
| easy | 24 | 0.917 | 0.933 | 1.000 | 1.000 | 7058 |
| hard | 12 | 0.833 | 0.889 | 1.000 | 0.000 | 6175 |
| medium | 24 | 0.870 | 0.903 | 0.000 | 0.000 | 4782 |

## Performance by patch source

| Source | N | Accuracy | Precision | Recall | F1 | PRRR | PFNRR |
|---|---|---|---|---|---|---|---|
| `gold` | 30 | 0.933 | 0.966 | 0.966 | 0.966 | 0.000 | 0.000 |
| `synthetic_expand_scope` | 8 | 0.875 | 1.000 | 0.875 | 0.933 | 0.000 | 1.000 |
| `synthetic_invert_conditional` | 7 | 0.857 | 0.500 | 1.000 | 0.667 | 1.000 | 0.000 |
| `synthetic_remove_addition` | 8 | 0.857 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `synthetic_remove_critical_line` | 7 | 0.714 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |

### Detection by mutation type

For each synthetic mutation, **detection rate** is the share of ground-truth `not_resolved` cases that MAS-Ops correctly rejected. Mutations whose harness label remained `resolved` are tracked separately — they exercise scope/style discipline, not correctness.

| Mutation source | N | GT not_resolved | GT resolved | Detection rate | Approve-rate when still resolved |
|---|---|---|---|---|---|
| `synthetic_expand_scope` | 8 | 0 | 8 | 0.000 | 0.875 |
| `synthetic_invert_conditional` | 7 | 6 | 1 | 0.833 | 1.000 |
| `synthetic_remove_addition` | 8 | 7 | 0 | 0.857 | 0.000 |
| `synthetic_remove_critical_line` | 7 | 6 | 1 | 0.833 | 0.000 |

## Variability across repetitions

- Instances with ≥2 repetitions: **30**
- Instances where repetitions disagreed on the decision: **18** (60.0%)
- Mean token stdev within instance: **4574**

## Qualitative observations

_Qualitative observations unavailable (ANTHROPIC_API_KEY is not set)._

## Notable cases

_Notable-cases narrative unavailable (ANTHROPIC_API_KEY is not set)._

### Top failures

- `django__django-15368` rep=1 (FN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-django__django-15368-rep1-synthetic_expand_scope-94a565c0`
- `django__django-13128` rep=1 (FP, decided_by=detective): `execution_id=rodada-2-django__django-13128-rep1-synthetic_remove_critical_line-06a15940`
- `django__django-13212` rep=1 (FN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-django__django-13212-rep1-gold-5da8a92d`
- `django__django-16032` rep=1 (FN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-django__django-16032-rep1-synthetic_remove_critical_line-e15ddf1c`
- `django__django-16502` rep=1 (FP, decided_by=detective): `execution_id=rodada-2-django__django-16502-rep1-synthetic_invert_conditional-8b505a2a`

### Top successes

- `astropy__astropy-8707` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-astropy__astropy-8707-rep1-synthetic_remove_addition-38aeae81`
- `django__django-12193` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-django__django-12193-rep1-synthetic_invert_conditional-9f5f9228`
- `django__django-13837` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-django__django-13837-rep1-synthetic_invert_conditional-ae3159c3`
- `scikit-learn__scikit-learn-13439` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-scikit-learn__scikit-learn-13439-rep1-synthetic_remove_critical_line-ba45c163`
- `sympy__sympy-14711` rep=1 (TN, decided_by=fixer_guardian_loop): `execution_id=rodada-2-sympy__sympy-14711-rep1-synthetic_invert_conditional-a90bfc01`

## Charts

- `charts/confusion_matrix.png`
- `charts/tokens_by_difficulty.png`
- `charts/accuracy_by_difficulty.png`
- `charts/guardian_iterations.png`

## Known limitations

- `tokens_by_agent` only covers agents using `llm.client.call_llm` (detective, fixer, fixer_guardian, executor_guardian). Executor and Communicator use the Claude Agent SDK and `langchain.create_agent` respectively, and their token usage must be retrieved from Langfuse traces on EC2-mas-ops.
- Execution is strictly sequential due to a known concurrency limitation in MAS-Ops. Rodada wall-clock time scales linearly with N × repetitions.
