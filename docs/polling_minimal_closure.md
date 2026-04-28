# Polling Minimal Closure

Date: 2026-04-28

## Goal

Add the smallest credible polling benchmark loop required by
`experiment_spec_codex_draft.md`:

- polling environment;
- polling AMQ feature map;
- BVI / AMQ / NNQ smoke configs;
- comparison runner;
- tests for the core polling dynamics.

## Implemented

- Environment:
  - `src/adversarial_queueing/envs/polling.py`
- Feature map:
  - `src/adversarial_queueing/features/polling_features.py`
- Configs:
  - `configs/polling_smoke.yaml`
  - `configs/polling_amq_smoke.yaml`
  - `configs/polling_nnq_smoke.yaml`
  - `configs/polling_comparison_smoke.yaml`
- Runner:
  - `scripts/run_polling_comparison.py`
- Tests:
  - `tests/test_polling.py`
  - `tests/test_polling_features.py`
  - `tests/test_polling_comparison_runner.py`

## Modeling Choices

State is represented as:

`(x_0, ..., x_{n-1}, p)`

where `p` is the current server position.

Nominal or defended polling selects one of the longest queues. A successful
attack, `attacker_action == 1` and `defender_action == 0`, redirects the polling
target to one of the shortest queues. Ties are split uniformly.

Switching cost is charged in expectation over tied polling targets. The service
completion event applies to the selected polling target after the polling
decision is resolved.

The smoke implementation uses uniformized CTMC transitions, matching the routing
and service-rate-control conventions.

## Smoke Result

Command:

```bash
rtk python scripts/run_polling_comparison.py --config configs/polling_comparison_smoke.yaml
```

Latest run:

`results/polling_smoke_comparison/20260428T144648Z`

Average cost:

- BVI: `0.422117`
- AMQ: `0.487983`
- NNQ: `0.294500`

This is a smoke result only. It proves the benchmark loop runs, not that the
polling ranking is meaningful.

## Policy Inspection

Polling smoke runs now export `policy_inspection.jsonl` and a compact
`policy_inspection` summary. The latest smoke comparison shows:

| Method | Mean p_defend | States p_defend >= 0.5 | Gap states p_defend >= 0.5 |
|---|---:|---:|---:|
| BVI | `0.291427` | `12 / 32` | `12 / 24` |
| AMQ | `0.000000` | `0 / 32` | `0 / 24` |
| NNQ | `1.000000` | `32 / 32` | `24 / 24` |

Interpretation:

- BVI has a nontrivial gap-dependent defense policy.
- AMQ smoke currently degenerates to never defending.
- NNQ smoke currently degenerates to always defending.

Therefore the NNQ smoke cost should not be read as a meaningful polling
performance result. The next polling step should inspect and calibrate these
policy shapes before increasing budgets.

## Verification

```bash
rtk python -m py_compile src/adversarial_queueing/envs/polling.py src/adversarial_queueing/features/polling_features.py src/adversarial_queueing/algorithms/amq.py src/adversarial_queueing/utils/config.py src/adversarial_queueing/evaluation/rollout.py scripts/run_experiment.py scripts/run_polling_comparison.py
rtk python -m unittest tests/test_polling.py tests/test_polling_features.py tests/test_polling_policy_inspection.py tests/test_polling_comparison_runner.py tests/test_rollout_evaluation.py tests/test_amq.py tests/test_nnq.py
```

## Next Steps

The next polling work should stay minimal:

1. Add a small polling policy-shape diagnostic report builder.
2. Calibrate AMQ/NNQ smoke settings enough to avoid always-defend /
   never-defend degeneracy.
3. Only then add a 3-seed polling debug comparison.
