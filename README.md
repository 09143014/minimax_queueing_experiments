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
python scripts/run_service_rate_comparison.py --config configs/service_rate_comparison.yaml
python -m unittest discover -s tests
```

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
