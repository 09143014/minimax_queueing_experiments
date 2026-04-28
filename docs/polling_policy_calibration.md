# Polling Policy Calibration

Date: 2026-04-28

## Goal

Move polling beyond the initial smoke closure by reducing obvious AMQ/NNQ policy
degeneracy before any multiseed comparison.

Initial smoke diagnostic:

`results/polling_policy_shape_diagnostic.md`

showed:

- BVI: state-dependent defense, `12 / 24` gap states defended;
- AMQ: never defends;
- NNQ: always defends.

## Calibration Attempts

### AMQ

AMQ was tested with two feature families and several small budgets:

| Config | Result |
|---|---|
| `configs/polling_amq_smoke.yaml` | never-defend |
| `configs/polling_amq_basic_200_smoke.yaml` | always-defend |
| `configs/polling_amq_basic_300_smoke.yaml` | always-defend |
| `configs/polling_amq_basic_500_smoke.yaml` | always-defend |
| `configs/polling_amq_basic_1000_smoke.yaml` | always-defend |
| `configs/polling_amq_basic_low_lr_1000_smoke.yaml` | never-defend |
| `configs/polling_amq_action_interaction_100_smoke.yaml` | always-defend |
| `configs/polling_amq_action_interaction_300_smoke.yaml` | always-defend |
| `configs/polling_amq_action_interaction_500_smoke.yaml` | always-defend |
| `configs/polling_amq_calibrated_smoke.yaml` | always-defend |

Conclusion: current polling AMQ update/features still jump between two
degenerate policies. Increasing budget or using action-interaction features
does not yet produce a BVI-like state-dependent policy.

### NNQ

NNQ calibration was more productive:

| Config | Avg cost | Mean p_defend | Gap states defended |
|---|---:|---:|---:|
| `configs/polling_nnq_smoke.yaml` | `0.294500` | `1.000000` | `24 / 24` |
| `configs/polling_nnq_calibrated_smoke.yaml` | `0.462333` | `0.068566` | `1 / 24` |
| `configs/polling_nnq_3000_smoke.yaml` | `0.447925` | `0.263625` | `0 / 24` |
| `configs/polling_nnq_augmented_3000_smoke.yaml` | `0.312392` | `0.491413` | `12 / 24` |

The useful change is `state_feature_set: polling_augmented`, which gives NNQ
explicit queue-gap, position one-hot, current-queue, and min/max indicators.

## Current Best Diagnostic Comparison

Command:

```bash
rtk python scripts/run_polling_comparison.py --config configs/polling_augmented_comparison_smoke.yaml
rtk python scripts/diagnose_polling_policy_shape.py --summary results/polling_augmented_smoke_comparison/20260428T155107Z/summary.json --json-output results/polling_augmented_policy_shape_diagnostic.json --markdown-output results/polling_augmented_policy_shape_diagnostic.md
```

Artifacts:

- `results/polling_augmented_smoke_comparison/20260428T155107Z/summary.json`
- `results/polling_augmented_policy_shape_diagnostic.json`
- `results/polling_augmented_policy_shape_diagnostic.md`

Result:

| Method | Avg cost | Mean p_defend | Gap states defended | Shape |
|---|---:|---:|---:|---|
| BVI | `0.422117` | `0.291427` | `12 / 24` | state-dependent |
| AMQ | `0.294500` | `1.000000` | `24 / 24` | always-defend |
| NNQ augmented | `0.312392` | `0.491413` | `12 / 24` | state-dependent |

## Interpretation

Polling is now partially calibrated:

- NNQ has a credible non-degenerate policy-shape candidate.
- AMQ remains degenerate and should not be promoted to a polling result.
- Smoke average cost is still not a final performance metric because AMQ and
  NNQ both over-defend zero-gap states under some configurations.

## Next Step

The next minimal polling step should target AMQ specifically. A good candidate is
a fitted Bellman calibration pass over bounded polling states, analogous to the
routing fitted calibration diagnostic, because plain online AMQ jumps between
never-defend and always-defend under the current smoke setup.
