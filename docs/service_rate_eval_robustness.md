# Service-Rate Evaluation Robustness

Date: 2026-04-28

## Goal

Check whether the service-rate-control ranking is an artifact of the short
debug evaluation protocol. The training budgets are unchanged; only the rollout
evaluation episode count is increased from `10` to `50`.

Methods compared:

- BVI
- AMQ
- raw NNQ
- NNQ+state0 guard

## Commands

Eval10 source:

```bash
rtk python scripts/run_service_rate_comparison_multiseed.py --config configs/service_rate_comparison_multiseed.yaml
```

Eval50 run:

```bash
rtk python scripts/run_service_rate_comparison_multiseed.py --config configs/service_rate_comparison_multiseed_eval50_debug.yaml
rtk python scripts/build_service_rate_report.py --summary results/service_rate_control_multiseed_eval50_debug_comparison/20260428T130213Z/summary.json --json-output results/service_rate_control_eval50_report.json --markdown-output results/service_rate_control_eval50_report.md
```

## Artifacts

Eval10 comparison:

`results/service_rate_control_multiseed_debug_comparison/20260428T124024Z/summary.json`

Eval50 comparison:

`results/service_rate_control_multiseed_eval50_debug_comparison/20260428T130213Z/summary.json`

Eval50 report:

- `results/service_rate_control_eval50_report.json`
- `results/service_rate_control_eval50_report.md`

## Mean Cost Comparison

| Method | Eval10 mean | Eval10 std | Eval50 mean | Eval50 std |
|---|---:|---:|---:|---:|
| BVI | `0.317933` | `0.020260` | `0.242748` | `0.000790` |
| AMQ | `0.363171` | `0.021307` | `0.293975` | `0.008094` |
| NNQ | `0.517298` | `0.018778` | `0.460424` | `0.063989` |
| NNQ+state0 guard | `0.392079` | `0.086671` | `0.317562` | `0.106242` |

Best-count by average cost is unchanged:

| Method | Eval10 best-count | Eval50 best-count |
|---|---:|---:|
| BVI | `1` | `1` |
| AMQ | `0` | `0` |
| NNQ | `0` | `0` |
| NNQ+state0 guard | `2` | `2` |

## Interpretation

The mean ranking is stable when increasing evaluation episodes:

1. BVI
2. AMQ
3. NNQ+state0 guard
4. raw NNQ

The eval50 run strengthens the conclusion that BVI and AMQ are stable service-
rate baselines. BVI's across-seed standard deviation drops sharply under more
evaluation episodes, and AMQ remains consistently between BVI and guarded NNQ.

The guarded NNQ result remains useful but volatile. It wins two of three seeds
under both eval10 and eval50, but its mean remains worse than AMQ because seed 2
is much worse than BVI and AMQ. This supports keeping NNQ+state0 guard as a
diagnostic/corrected baseline rather than promoting it to the main NNQ result.

## Conclusion

The service-rate-control result is robust enough for the current debug-stage
narrative:

- raw NNQ is weak because it over-services at the empty queue;
- the state0 guard repairs much of that failure;
- the guard does not beat AMQ or BVI on mean cost;
- BVI remains the strongest mean-cost method under eval50;
- AMQ remains the strongest learned non-oracle-style method among the current
  service-rate baselines.
