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
- NNQ Bellman residual mean: `0.6447`
- NNQ-vs-bounded-BVI-reference Q gap mean: `7.4908`
- mean absolute NNQ Q: `0.5351`
- mean action Q spread: `0.2020`
- inspected bounded routing states: `64`

The 200-step smoke run is useful as an interface check. It learns a degenerate
all-no-defend policy on the bounded inspection grid. The Q diagnostic shows that
the learned Q scale is far below the bounded BVI reference scale, so it is not
acceptable as a final routing NNQ baseline.

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
- NNQ Bellman residual mean: `0.1852`
- NNQ-vs-bounded-BVI-reference Q gap mean: `2.1684`
- mean absolute NNQ Q: `5.8576`
- mean action Q spread: `0.2554`
- inspected bounded routing states: `64`

The debug run confirms that routing NNQ can produce nontrivial defense
probabilities, and the Bellman/Q diagnostic shows much better Q scale than the
smoke run. However, the policy is still worse than the smoke path by the bounded
BVI-reference policy diagnostic and has higher rollout cost. The remaining issue
appears to be weak action separation rather than only global Q under-scaling.
This should not be used as the final NNQ result. The next NNQ step should
improve action discrimination or trainer stability before running larger
multi-seed comparisons.

## Action-Separation Debug

The first action-separation fix adds `state_feature_set: routing_augmented` for
routing NNQ. The augmented state is still environment-only information: queue
lengths, service rates, queue/service-rate ratios, total load, min/max load,
imbalance, and min/max queue indicators. It does not use BVI values, BVI
policies, or bounded reference diagnostics during training.

Command:

```bash
rtk python scripts/run_experiment.py --config configs/routing_nnq_debug.yaml
```

Artifact:

```text
results/routing_nnq_debug/20260427T114634Z
```

Summary:

- rollout average cost mean: `0.1926`
- bounded policy comparison mean absolute defend-probability gap: `0.1454`
- NNQ over-defense states: `0`
- NNQ under-defense states: `10`
- NNQ Bellman residual mean: `0.1400`
- NNQ-vs-bounded-BVI-reference Q gap mean: `2.0613`
- mean absolute NNQ Q: `5.9647`
- mean action Q spread: `0.3977`
- inspected bounded routing states: `64`

This improves action Q spread and Bellman consistency relative to the first
debug path, without introducing over-defense. It still learns an all-no-defend
policy on the bounded inspection grid, so it is not final. The next useful NNQ
step is a targeted policy-calibration fix, not another blind increase in step
count.

Rejected variants:

- `routing_augmented`, hidden size `32`, balanced action batches: policy gap
  worsened to `0.6872` with `39` over-defense states.
- `routing_augmented`, hidden size `64`, no balanced batches: policy gap
  worsened to `0.3946` with `15` over-defense states.
