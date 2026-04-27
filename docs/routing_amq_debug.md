# Routing AMQ Debug Notes

This note records the first non-smoke AMQ debugging configuration for the routing benchmark.

## Commands

```bash
rtk python3 scripts/run_experiment.py --config configs/routing_amq_debug.yaml
rtk python3 scripts/run_routing_amq_multiseed.py --config configs/routing_amq_multiseed_debug.yaml
rtk python3 scripts/run_experiment.py --config configs/routing_smoke.yaml
rtk python3 -m unittest discover -s tests -v
```

## Current Debug Configuration

- Environment: 3 parallel queues.
- Arrival rate: `2.0`.
- Service rates: `[1.0, 1.5, 2.0]`.
- Feature set: `action_interaction`.
- AMQ steps: `50000`.
- Defense cost: `0.5`.
- AMQ step size: constant `eta0=0.001`.
- Exploring starts: probability `0.1`, bounded grid `0..3` per queue.
- Evaluation: `10` episodes, horizon `50`, seed `300`.
- Policy inspection: bounded grid `0..3` per queue, defend-probability threshold `0.5`.

## Interpretation

The routing AMQ debug run is intended to test whether the learned defender policy starts using defense selectively across queue configurations. It is not a final performance claim. Compare it against `configs/routing_smoke.yaml`, which runs bounded BVI on the same small grid and exports `policy_inspection.jsonl`.

## Observed Behavior

The first debug run with `action_interaction` features and no exploring starts was non-crashing and exported the expected metrics, rollout rows, and policy inspection rows, but the learned policy was degenerate:

- AMQ chooses defend with probability `1.0` on all inspected states.
- BVI on the small routing grid chooses defense selectively, with some low-load states not defended.
- The policy comparison diagnostic reports the AMQ-vs-bounded-BVI defend-probability gap by state, total queue length, and imbalance.
- The Q diagnostic reports AMQ Bellman residuals and AMQ-vs-bounded-BVI-reference Q gaps for every inspected `(state, attacker_action, defender_action)` entry.

Adding bounded exploring starts improves the seed-0 debug policy substantially:

- AMQ no longer chooses defend everywhere.
- The seed-0 policy gap against bounded BVI drops sharply compared with the no-exploring-starts debug run.
- A short multi-seed probe still shows variance, so this is a better debug configuration, not a final result claim.
- Use `configs/routing_amq_multiseed_debug.yaml` to track this variance explicitly across configured seeds.

The current 3-seed debug run reports:

- mean rollout average cost about `0.266`;
- mean defend-probability policy gap about `0.215`;
- mean AMQ Bellman residual about `0.154`;
- mean AMQ-vs-bounded-BVI-reference Q gap about `0.408`;
- persistent under-defense on a small set of low-queue states where bounded BVI uses mixed defense.

## Follow-Up

- Compare AMQ and BVI policy inspection rows by queue imbalance and total queue length.
- Run multiple seeds before interpreting policy quality.
- Tune `defend_cost` and `attack_cost` if the defense policy is degenerate.
- Use the Q diagnostic to track whether training changes reduce AMQ Bellman residuals without increasing AMQ-vs-bounded-BVI-reference Q gaps.
- Keep BVI labeled as a bounded-state approximate solver, not ground truth.
