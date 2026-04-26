# Service-Rate-Control NNQ Debug Notes

This note records the first NNQ debugging configuration for the service-rate-control benchmark.

## Implementation Status

The current NNQ implementation is a NumPy MLP smoke baseline because PyTorch is not available in the current execution environment. It is useful for validating the shared NNQ pipeline:

- replay buffer;
- target network;
- Adam-style optimizer;
- epsilon exploration;
- Q-matrix output for finite attacker/defender actions;
- minimax target using the shared matrix-game solver;
- shared rollout evaluation and policy-grid export.

It should not yet be treated as the final strong NNQ baseline required for publication experiments.

## Commands

```bash
rtk python3 scripts/run_experiment.py --config configs/nnq_debug.yaml
rtk python3 -m unittest discover -s tests -v
```

## Current Debug Configuration

- Environment: single-queue service-rate-control.
- Hidden size: `32`.
- NNQ steps: `3000`.
- Batch size: `32`.
- Replay capacity: `5000`.
- Target update interval: `250`.
- Epsilon: `0.2`.
- Evaluation: `10` episodes, horizon `50`, seed `200`.
- Policy grid: states `0..20`, high-probability threshold `0.5`.

## Observed Behavior

The NNQ debug run produces a conservative policy:

- states `0..20`: high service rate with probability `1.0`.

This is a valid smoke/debug signal because the NNQ path trains, evaluates, and exports policy grids end-to-end. It is not yet a satisfactory learned baseline because BVI chooses low service rate at state `0`, while NNQ chooses high service rate everywhere.

## Follow-Up

- If PyTorch becomes available, replace or supplement this NumPy smoke baseline with a stronger PyTorch NNQ implementation.
- Tune learning rate, target update interval, state scaling, and exploration before comparing NNQ to AMQ/BVI.
- Add multi-seed debug runs before interpreting NNQ policy quality.
- Keep this implementation labeled as `numpy_mlp_smoke` in result summaries.

