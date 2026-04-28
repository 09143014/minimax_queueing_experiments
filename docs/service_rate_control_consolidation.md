# Service-Rate-Control Consolidation

Date: 2026-04-28

## Goal

Bring the service-rate-control benchmark closer to the routing experiment
structure: BVI / AMQ / NNQ comparison, multiseed aggregation, and a reproducible
report artifact.

## Implemented

- Added multiseed comparison runner:
  - `scripts/run_service_rate_comparison_multiseed.py`
- Added configs:
  - `configs/service_rate_comparison_multiseed_smoke.yaml`
  - `configs/service_rate_comparison_multiseed.yaml`
- Added report builder:
  - `scripts/build_service_rate_report.py`
- Added policy-shape diagnostic:
  - `scripts/diagnose_service_rate_policy_shape.py`
- Added tests:
  - `tests/test_service_rate_comparison_multiseed_runner.py`
  - `tests/test_service_rate_report.py`
  - `tests/test_service_rate_policy_shape_diagnostic.py`
- Updated README Quick Start with service-rate multiseed and report commands.

## Commands Run

```bash
rtk python scripts/run_service_rate_comparison_multiseed.py --config configs/service_rate_comparison_multiseed.yaml
rtk python scripts/build_service_rate_report.py --summary results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json --json-output results/service_rate_control_report.json --markdown-output results/service_rate_control_report.md
rtk python scripts/diagnose_service_rate_policy_shape.py --summary results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json --json-output results/service_rate_policy_shape_diagnostic.json --markdown-output results/service_rate_policy_shape_diagnostic.md
rtk python -m py_compile scripts/run_service_rate_comparison.py scripts/run_service_rate_comparison_multiseed.py scripts/build_service_rate_report.py scripts/diagnose_service_rate_policy_shape.py src/adversarial_queueing/envs/service_rate_control.py src/adversarial_queueing/features/service_rate_features.py
rtk python -m unittest tests/test_service_rate_control.py tests/test_service_rate_features.py tests/test_service_rate_comparison_runner.py tests/test_service_rate_comparison_multiseed_runner.py tests/test_service_rate_report.py tests/test_service_rate_policy_shape_diagnostic.py tests/test_policy_grid.py
```

All checks passed.

## Result

Result file:

`results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json`

Average cost mean over 3 seeds:

- BVI: `0.317933`
- AMQ: `0.363171`
- NNQ: `0.517298`

Ranking counts:

- BVI: 3 seeds
- AMQ: 0 seeds
- NNQ: 0 seeds

Report files:

- `results/service_rate_control_report.json`
- `results/service_rate_control_report.md`

Policy-shape diagnostic files:

- `results/service_rate_policy_shape_diagnostic.json`
- `results/service_rate_policy_shape_diagnostic.md`

## Current Interpretation

BVI is strongest on the current service-rate-control debug comparison. AMQ is
consistently better than NNQ, but remains behind BVI. NNQ's weak result appears
to be an empty-state over-service problem: on all three seeds, NNQ assigns high
service probability at state 0, while BVI uses low service and AMQ uses medium
service at state 0.

This is a consolidation result, not a final publishable service-rate-control
benchmark result. The next service-rate step should inspect whether NNQ needs a
stronger configuration, policy calibration, or a different budget. Do not change
the service-rate-control environment dynamics based on this diagnostic.

Follow-up minimal NNQ experiments are recorded in:

`docs/service_rate_nnq_minimal_experiments.md`

Summary of that follow-up: forced low/medium behavior, uniform exploration,
state-scale adjustment, longer training, and lower learning rate all failed to
fix state 0 high-service overuse. This makes blind NNQ config tuning a poor next
step.

Repair diagnostic:

`docs/service_rate_nnq_repair_diagnostic.md`

The 3-seed repair diagnostic shows that changing only the rollout policy at
`state == 0` from NNQ's high service to low service improves mean NNQ cost from
`0.517298` to `0.392079`. Broader low-state overrides are worse, so the useful
signal is narrow: empty-state over-service is real, but broad benchmark-specific
policy overrides should not be folded into raw NNQ.

Unified comparison update:

`results/service_rate_control_multiseed_debug_comparison/20260428T124024Z/summary.json`

The service-rate comparison runner now includes a fourth method,
`nnq_state0_guard`, which trains the same NNQ baseline but evaluates it with the
documented `state == 0 -> low service` guard. In the 3-seed debug comparison,
mean costs are:

- BVI: `0.317933`
- AMQ: `0.363171`
- NNQ: `0.517298`
- NNQ+state0 guard: `0.392079`

This confirms the repair closes much of the NNQ gap while preserving the main
ranking by mean cost: BVI, AMQ, guarded NNQ, raw NNQ. The guard has high
per-seed variability and should remain a diagnostic/corrected baseline rather
than the raw NNQ baseline.
