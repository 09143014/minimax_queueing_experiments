# Service-Rate NNQ Minimal Experiments

Date: 2026-04-28

## Goal

Investigate the service-rate-control NNQ failure mode identified in
`results/service_rate_policy_shape_diagnostic.md`: NNQ over-services at the
empty queue by choosing high service at state 0 on every seed.

The acceptance criterion for a useful minimal experiment was:

- state 0 should no longer choose high service with probability `1.0`; and
- average cost should improve against the current NNQ baseline
  (`0.502686` on seed 0).

## Baseline

Config:

`configs/nnq_debug.yaml`

Seed 0 result:

- run: `results/service_rate_control_nnq_debug/20260428T093718Z`
- average cost: `0.502686`
- state 0 policy: low `0.0`, medium `0.0`, high `1.0`

BVI state 0 reference Q diagnostic:

- low column is best under the BVI minimax matrix
- high column is more costly at state 0

NNQ state 0 Q diagnostic showed the opposite ordering: NNQ assigns the high
service column the lowest minimax-relevant value.

## Negative Config Experiments

### Forced medium behavior

Config:

`configs/nnq_forced_medium_debug.yaml`

Run:

`results/service_rate_control_nnq_forced_medium_debug/20260428T101136Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`

Conclusion: increasing medium-action behavior coverage does not fix the learned
policy.

### Forced low behavior

Config:

`configs/nnq_forced_low_debug.yaml`

Run:

`results/service_rate_control_nnq_forced_low_debug/20260428T101314Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`

Conclusion: increasing low-action behavior coverage does not fix the learned
policy.

### Uniform exploration

Config:

`configs/nnq_uniform_explore_debug.yaml`

Run:

`results/service_rate_control_nnq_uniform_explore_debug/20260428T101447Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`

Conclusion: full random action-pair behavior does not fix the learned policy.

### Longer training

Config:

`configs/nnq_long_debug.yaml`

Run:

`results/service_rate_control_nnq_long_debug/20260428T102212Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`
- final loss increased sharply

Conclusion: simply increasing NNQ steps from `3000` to `10000` is not useful.

### State scale 1.0

Config:

`configs/nnq_state_scale1_debug.yaml`

Run:

`results/service_rate_control_nnq_state_scale1_debug/20260428T102038Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`

Conclusion: making the network input more sensitive to small queue lengths does
not fix the learned policy.

### Lower learning rate

Config:

`configs/nnq_low_lr_debug.yaml`

Run:

`results/service_rate_control_nnq_low_lr_debug/20260428T102624Z`

Result:

- average cost: `0.502686`
- state 0 policy: high `1.0`
- final loss remained large

Conclusion: reducing learning rate alone does not fix the failure mode.

## Rejected Structural Attempts

Two structural ideas were tried locally and removed because they did not improve
the seed 0 result:

- service-rate augmented NNQ state features `[x, x^2, is_zero]`
- NNQ fitted Bellman self-calibration over bounded service-rate states

Both still selected high service at state 0 and kept average cost at `0.502686`.
They are not retained in the codebase.

## Current Conclusion

The service-rate NNQ failure is robust to small behavior-policy, exploration,
learning-rate, and budget changes. The issue appears to be a systematic Q
ordering error at state 0, not a simple lack of action coverage.

Recommended next step:

Do not continue blind NNQ config tuning. If NNQ must be strengthened for
service-rate-control, use a more deliberate algorithmic change, such as a
separate tabular low-state correction, a better target-stabilization design, or
a stronger neural implementation with validation against policy-shape
diagnostics.
