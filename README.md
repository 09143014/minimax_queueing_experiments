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
- shared rollout evaluation for BVI and AMQ smoke runs;
- policy-grid export for service-rate threshold inspection;
- tests for the minimax solver and service-rate-control dynamics.

Routing, polling, AMQ, and NNQ will be added in focused follow-up changes.

## Quick Start

```bash
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/amq_smoke.yaml
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
