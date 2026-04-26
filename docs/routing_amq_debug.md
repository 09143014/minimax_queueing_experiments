# Routing AMQ Debug Notes

This note records the first non-smoke AMQ debugging configuration for the routing benchmark.

## Commands

```bash
rtk python3 scripts/run_experiment.py --config configs/routing_amq_debug.yaml
rtk python3 scripts/run_experiment.py --config configs/routing_smoke.yaml
rtk python3 -m unittest discover -s tests -v
```

## Current Debug Configuration

- Environment: 3 parallel queues.
- Arrival rate: `2.0`.
- Service rates: `[1.0, 1.5, 2.0]`.
- Feature set: `action_interaction`.
- AMQ steps: `10000`.
- Defense cost: `0.5`.
- AMQ step size: Robbins-Monro with `eta0=0.01`, `decay_power=0.6`.
- Evaluation: `10` episodes, horizon `50`, seed `300`.
- Policy inspection: bounded grid `0..3` per queue, defend-probability threshold `0.5`.

## Interpretation

The routing AMQ debug run is intended to test whether the learned defender policy starts using defense selectively across queue configurations. It is not a final performance claim. Compare it against `configs/routing_smoke.yaml`, which runs bounded BVI on the same small grid and exports `policy_inspection.jsonl`.

## Observed Behavior

The first debug run with `action_interaction` features is non-crashing and exports the expected metrics, rollout rows, and policy inspection rows, but the learned policy is still degenerate:

- AMQ chooses defend with probability `1.0` on all inspected states.
- BVI on the small routing grid chooses defense selectively, with some low-load states not defended.
- The policy comparison diagnostic reports the AMQ-vs-bounded-BVI defend-probability gap by state, total queue length, and imbalance.
- This indicates the routing AMQ path is ready for debugging, but the current feature/training configuration should not be interpreted as a satisfactory learned routing policy.

## Follow-Up

- Compare AMQ and BVI policy inspection rows by queue imbalance and total queue length.
- Run multiple seeds before interpreting policy quality.
- Tune `defend_cost` and `attack_cost` if the defense policy is degenerate.
- Add a routing-specific behavior-cloning or Bellman residual diagnostic against BVI rows to see whether the issue is feature expressiveness, learning dynamics, or cost scale.
- Keep BVI labeled as a bounded-state approximate solver, not ground truth.
