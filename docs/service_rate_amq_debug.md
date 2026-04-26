# Service-Rate-Control AMQ Debug Notes

This note records the first non-smoke AMQ debugging configuration for the service-rate-control benchmark.

## Commands

```bash
rtk python3 scripts/run_experiment.py --config configs/amq_debug.yaml
rtk python3 scripts/run_experiment.py --config configs/bvi_debug.yaml
rtk python3 -m unittest discover -s tests -v
```

## Current Debug Configuration

- Environment: single-queue service-rate-control.
- Feature set: `action_interaction`.
- AMQ steps: `10000`.
- AMQ step size: Robbins-Monro with `eta0=0.01`, `decay_power=0.6`.
- Evaluation: `10` episodes, horizon `50`, seed `200`.
- Policy grid: states `0..20`, high-probability threshold `0.5`.

## Observed Behavior

The AMQ debug run produces a threshold-like defender policy:

- state `0`: mostly medium service rate;
- state `1`: high service rate with probability above `0.5`;
- states `2..20`: high service rate with probability `1.0`.

The BVI debug run produces:

- state `0`: low service rate;
- states `1..20`: high service rate.

This is not a final experimental claim. It only shows that the AMQ training path is no longer stuck in the short-smoke degenerate policy and that policy-grid inspection can diagnose threshold structure.

## Follow-Up

- Run multiple seeds before interpreting AMQ vs BVI differences.
- Add BVI truncation sensitivity before treating BVI as a strong reference.
- Tune service costs and attack cost if the threshold at state `1` is too degenerate for publication experiments.
- Keep BVI labeled as a bounded-state approximate solver, not ground truth.

