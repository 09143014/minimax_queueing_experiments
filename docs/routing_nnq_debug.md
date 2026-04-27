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

## Defender-Action Exploration

The next fix adds `forced_defender_action_probability` to the NNQ behavior
policy. This is training-only exploration: with a small probability the sampled
defender action is overwritten by a configured defender action. For routing
debug this action is `1` (defend). Evaluation still uses the learned minimax
NNQ policy, and BVI remains evaluation-only.

Command:

```bash
rtk python scripts/run_experiment.py --config configs/routing_nnq_debug.yaml
```

Artifact:

```text
results/routing_nnq_debug/20260427T122157Z
```

Summary for `forced_defender_action_probability: 0.1`:

- rollout average cost mean: `0.1861`
- bounded policy comparison mean absolute defend-probability gap: `0.1299`
- NNQ over-defense states: `1`
- NNQ under-defense states: `6`
- NNQ Bellman residual mean: `0.1606`
- NNQ-vs-bounded-BVI-reference Q gap mean: `2.0390`
- mean absolute NNQ Q: `5.9869`
- mean action Q spread: `0.3038`
- inspected bounded routing states: `64`

This is the best routing NNQ debug configuration so far by bounded policy gap:
it reduces under-defense relative to the augmented-feature debug run while
keeping over-defense limited to one inspected state. It is still not final,
because the one over-defense state is the empty state `[0, 0, 0]`.

Rejected exploration probabilities:

- `0.05`: policy gap worsened to `0.3668` with `17` over-defense states and
  `5` under-defense states.
- `0.15`: policy gap worsened to `0.1419` with `1` over-defense state and `8`
  under-defense states.
