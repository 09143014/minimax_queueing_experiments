# Minimax Queueing Experiments

Research simulation code for comparing NNQ, AMQ, and BVI on adversarial queueing/control Markov games.

This repository is intentionally conservative: benchmark dynamics, algorithms, experiment runners, and analysis should remain separate so comparisons are reproducible and inspectable.

## Current Status

The first runnable baseline implements:

- a shared environment interface;
- a service-rate-control benchmark using uniformized CTMC dynamics;
- a shared zero-sum matrix-game solver;
- a bounded value-iteration smoke path;
- a linear AMQ smoke path for service-rate-control;
- a NumPy MLP NNQ smoke path for service-rate-control;
- an initial parallel-queue routing benchmark environment and feature map;
- BVI and linear AMQ smoke paths for routing;
- routing comparison runners for BVI / AMQ / NNQ, including multi-seed aggregation;
- normalized routing AMQ feature experiments for calibration;
- shared rollout evaluation for BVI and AMQ smoke runs across implemented benchmarks;
- policy-grid export for service-rate threshold inspection;
- policy inspection export for routing defender decisions;
- BVI truncation sensitivity checks for service-rate-control;
- tests for the minimax solver and service-rate-control dynamics.

Polling and broader multi-benchmark experiment runners will be added in focused follow-up changes.

## Quick Start

```bash
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/amq_smoke.yaml
python scripts/run_experiment.py --config configs/nnq_smoke.yaml
python scripts/run_experiment.py --config configs/routing_smoke.yaml
python scripts/run_experiment.py --config configs/routing_amq_smoke.yaml
python scripts/run_experiment.py --config configs/routing_amq_debug.yaml
python scripts/run_routing_amq_multiseed.py --config configs/routing_amq_multiseed_debug.yaml
python scripts/run_routing_comparison.py --config configs/routing_comparison_debug.yaml
python scripts/run_routing_comparison_multiseed.py --config configs/routing_comparison_multiseed_debug.yaml
python scripts/run_routing_comparison_multiseed.py --config configs/routing_comparison_normalized_amq_multiseed_eval50_debug.yaml
python scripts/run_service_rate_comparison.py --config configs/service_rate_comparison.yaml
python scripts/run_service_rate_comparison_multiseed.py --config configs/service_rate_comparison_multiseed.yaml
python -m unittest discover -s tests
```

For the current routing narrative result, rebuild the comparison and diagnostic
artifacts from existing summaries:

```bash
python scripts/build_routing_comparison_report.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --normalized-visitation-summary results/routing_amq_normalized_visitation_seed0_debug/20260427T170257Z/summary.json --fitted-visitation-summary results/routing_amq_normalized_fitted_debug/20260427T180135Z/summary.json --json-output results/routing_comparison_report.json --markdown-output results/routing_comparison_report.md
python scripts/diagnose_routing_amq_cost_loss.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --json-output results/routing_amq_cost_loss_diagnostic.json --markdown-output results/routing_amq_cost_loss_diagnostic.md
python scripts/diagnose_routing_bvi_attacker_visitation.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --json-output results/routing_bvi_attacker_visitation_diagnostic.json --markdown-output results/routing_bvi_attacker_visitation_diagnostic.md
python scripts/build_routing_narrative_report.py --comparison-report results/routing_comparison_report.json --cost-loss-diagnostic results/routing_amq_cost_loss_diagnostic.json --bvi-attacker-visitation-diagnostic results/routing_bvi_attacker_visitation_diagnostic.json --json-output results/routing_narrative_report.json --markdown-output results/routing_narrative_report.md
```

The current routing conclusion is intentionally narrow: normalized AMQ is the
selected AMQ performance baseline, while fitted AMQ is a calibration diagnostic.
Fitted calibration improves policy-shape diagnostics but reduces defense mass on
states visited by the common BVI attacker, so it is not the selected performance
result.

For the current service-rate-control result, rebuild the report from the latest
3-seed debug comparison:

```bash
python scripts/build_service_rate_report.py --summary results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json --json-output results/service_rate_control_report.json --markdown-output results/service_rate_control_report.md
python scripts/diagnose_service_rate_policy_shape.py --summary results/service_rate_control_multiseed_debug_comparison/20260428T093705Z/summary.json --json-output results/service_rate_policy_shape_diagnostic.json --markdown-output results/service_rate_policy_shape_diagnostic.md
```

The current service-rate-control conclusion is also narrow: BVI is strongest on
the debug comparison, AMQ is consistently better than NNQ, and NNQ's weak result
appears to come from empty-state over-service rather than a benchmark dynamics
issue.

If `pytest` is installed, the tests are also pytest-compatible:

```bash
pytest
```

## Research Conventions

- Costs are defender costs and attacker rewards.
- Defender minimizes cost; attacker maximizes the same cost.
- Matrix-game payoffs use shape `[num_attacker_actions, num_defender_actions]`.
- BVI is a bounded-state approximate solver, not hard-coded ground truth.
- Service-rate-control defaults to three service levels and quadratic congestion cost.
