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
- Added tests:
  - `tests/test_service_rate_comparison_multiseed_runner.py`
  - `tests/test_service_rate_report.py`
- Updated README Quick Start with service-rate multiseed and report commands.

## Commands Run

```bash
rtk python scripts/run_service_rate_comparison_multiseed.py --config configs/service_rate_comparison_multiseed.yaml
rtk python scripts/build_service_rate_report.py --summary results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json --json-output results/service_rate_control_report.json --markdown-output results/service_rate_control_report.md
rtk python -m py_compile scripts/run_service_rate_comparison.py scripts/run_service_rate_comparison_multiseed.py scripts/build_service_rate_report.py src/adversarial_queueing/envs/service_rate_control.py src/adversarial_queueing/features/service_rate_features.py
rtk python -m unittest tests/test_service_rate_control.py tests/test_service_rate_features.py tests/test_service_rate_comparison_runner.py tests/test_service_rate_comparison_multiseed_runner.py tests/test_service_rate_report.py tests/test_policy_grid.py
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

## Current Interpretation

BVI is strongest on the current service-rate-control debug comparison. AMQ is
consistently better than NNQ, but remains behind BVI. NNQ does not yet show the
high-service threshold behavior seen in BVI and AMQ.

This is a consolidation result, not a final publishable service-rate-control
benchmark result. The next service-rate step should inspect whether NNQ needs a
stronger configuration or whether the current debug budget is intentionally too
small.
