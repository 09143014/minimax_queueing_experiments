# Routing NNQ Debug Notes

This note records the first routing NNQ smoke/debug path. BVI is used only as a
bounded evaluation reference after NNQ training; no BVI values or policies are
used to train, initialize, normalize, or early-stop NNQ.

## Smoke Path

Command:

```bash
rtk python scripts/run_experiment.py --config configs/routing_nnq_smoke.yaml
```

Artifact:

```text
results/routing_nnq_smoke/20260427T101614Z
```

Summary:

- rollout average cost mean: `0.1047`
- bounded policy comparison mean absolute defend-probability gap: `0.1454`
- NNQ over-defense states: `0`
- NNQ under-defense states: `10`
- inspected bounded routing states: `64`

The 200-step smoke run is useful as an interface check. It learns a degenerate
all-no-defend policy on the bounded inspection grid, so it is not acceptable as
a final routing NNQ baseline.

## First Debug Path

Command:

```bash
rtk python scripts/run_experiment.py --config configs/routing_nnq_debug.yaml
```

Artifact:

```text
results/routing_nnq_debug/20260427T103712Z
```

Current debug config uses a smaller MLP, bounded routing exploring starts, and a
5000-step budget to keep the run usable during development.

Summary:

- rollout average cost mean: `0.1834`
- bounded policy comparison mean absolute defend-probability gap: `0.1951`
- NNQ over-defense states: `3`
- NNQ under-defense states: `10`
- inspected bounded routing states: `64`

The debug run confirms that routing NNQ can produce nontrivial defense
probabilities, but the policy is still worse than the smoke path by the bounded
BVI-reference diagnostic and has higher rollout cost. This should not be used as
the final NNQ result. The next NNQ step should diagnose target/Q scale and
Bellman consistency rather than blindly increasing training steps.
