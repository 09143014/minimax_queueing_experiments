# Current Handoff

Date: 2026-04-29

## Repository State

Branch/worktree state at handoff:

- latest commit: `dc40833 Calibrate polling policy smoke diagnostics`
- working tree: clean

Recent commits:

- `dc40833 Calibrate polling policy smoke diagnostics`
- `4a77f75 Add polling policy-shape diagnostic report`
- `39f81f0 Add polling policy inspection`
- `ae62328 Add polling benchmark smoke loop`
- `22cc277 Add service-rate eval50 robustness check`
- `4683995 Add service-rate guarded NNQ comparison`
- `4166488 Add service-rate NNQ repair diagnostic`
- `2bed77b Record service-rate NNQ minimal experiments`

## Overall Progress

The project now has runnable/debuggable loops for all three benchmarks from
`experiment_spec_codex_draft.md`:

- routing
- service-rate-control
- polling

Routing and service-rate-control have multi-seed comparison and narrative-level
diagnostics. Polling has a smoke loop, policy inspection, and policy-shape
diagnostics, but is not yet ready for multiseed performance claims.

## Routing Status

Routing is currently the most mature benchmark.

Implemented:

- BVI / AMQ / NNQ comparison runner
- multiseed runner
- normalized AMQ baseline
- fitted AMQ calibration diagnostic
- cost-loss diagnostic
- BVI-attacker visitation diagnostic
- narrative report

Key files:

- `scripts/run_routing_comparison.py`
- `scripts/run_routing_comparison_multiseed.py`
- `scripts/build_routing_comparison_report.py`
- `scripts/build_routing_narrative_report.py`
- `docs/routing_amq_calibration_handoff.md`
- `results/routing_narrative_report.md`

Current conclusion:

- normalized AMQ is the selected AMQ performance baseline;
- fitted AMQ improves policy-shape diagnostics but loses important BVI-attacker
  defense mass, so it remains diagnostic rather than selected performance result.

## Service-Rate-Control Status

Service-rate-control has a solid debug-stage comparison.

Implemented:

- BVI / AMQ / NNQ / NNQ+state0 guard comparison
- 3-seed eval10 comparison
- 3-seed eval50 robustness comparison
- NNQ minimal config diagnostics
- NNQ state0 repair diagnostic

Key files:

- `scripts/run_service_rate_comparison.py`
- `scripts/run_service_rate_comparison_multiseed.py`
- `scripts/build_service_rate_report.py`
- `scripts/diagnose_service_rate_policy_shape.py`
- `scripts/diagnose_service_rate_nnq_repair.py`
- `docs/service_rate_control_consolidation.md`
- `docs/service_rate_eval_robustness.md`
- `docs/service_rate_nnq_repair_diagnostic.md`

Important results:

- eval50 BVI mean: `0.242748`
- eval50 AMQ mean: `0.293975`
- eval50 NNQ+state0 guard mean: `0.317562`
- eval50 raw NNQ mean: `0.460424`

Current conclusion:

- raw NNQ is weak because it over-services at the empty queue;
- `state == 0 -> low service` repairs much of that failure;
- guarded NNQ remains diagnostic/corrected baseline, not raw NNQ;
- BVI and AMQ remain stronger by mean cost under eval50.

## Polling Status

Polling was added recently and is currently at calibrated smoke-diagnostic stage.

Implemented:

- polling environment
- polling AMQ feature map
- BVI / AMQ / NNQ smoke configs
- polling comparison runner
- policy inspection
- policy-shape diagnostic report
- NNQ `polling_augmented` state feature set

Key files:

- `src/adversarial_queueing/envs/polling.py`
- `src/adversarial_queueing/features/polling_features.py`
- `src/adversarial_queueing/evaluation/polling_policy.py`
- `scripts/run_polling_comparison.py`
- `scripts/diagnose_polling_policy_shape.py`
- `docs/polling_minimal_closure.md`
- `docs/polling_policy_calibration.md`

Current best polling diagnostic comparison:

`results/polling_augmented_smoke_comparison/20260428T155107Z/summary.json`

Policy-shape report:

`results/polling_augmented_policy_shape_diagnostic.md`

Current polling result:

| Method | Avg cost | Mean p_defend | Gap states defended | Shape |
|---|---:|---:|---:|---|
| BVI | `0.422117` | `0.291427` | `12 / 24` | state-dependent |
| AMQ | `0.294500` | `1.000000` | `24 / 24` | always-defend |
| NNQ augmented | `0.312392` | `0.491413` | `12 / 24` | state-dependent |

Interpretation:

- polling smoke loop is now runnable and diagnosable;
- NNQ has a credible non-degenerate polling candidate via
  `state_feature_set: polling_augmented`;
- AMQ remains degenerate, flipping between never-defend and always-defend under
  the tested smoke settings;
- polling smoke costs are not performance claims yet.

## Immediate Next Task

Do polling AMQ fitted Bellman calibration over bounded polling states.

Goal:

Determine whether polling AMQ degeneracy is caused by unstable online TD
training or by insufficient feature representation.

Minimal implementation direction:

1. Extend AMQ fitted calibration in `src/adversarial_queueing/algorithms/amq.py`
   to support `PollingEnv`, not only `RoutingEnv`.
2. Use bounded polling states of the form `(x0, x1, ..., position)`.
3. Add a polling fitted AMQ config, probably starting from:
   - `configs/polling_amq_action_interaction_100_smoke.yaml`
   - or `configs/polling_amq_calibrated_smoke.yaml`
4. Run smoke comparison with:
   - BVI: `configs/polling_smoke.yaml`
   - AMQ fitted polling candidate
   - NNQ augmented: `configs/polling_nnq_augmented_3000_smoke.yaml`
5. Rebuild polling policy-shape diagnostic.

Acceptance criterion for a useful AMQ calibration:

- AMQ should not be `never_defend` or `always_defend`;
- AMQ should defend some but not all gap states;
- ideally, AMQ should move toward BVI's `12 / 24` gap-state defense shape.

If fitted AMQ still degenerates:

- record it as a negative structural diagnostic;
- do not proceed to 3-seed polling comparison yet;
- next likely issue is polling feature design, not just training instability.

## Useful Commands

Current polling augmented comparison:

```bash
rtk python scripts/run_polling_comparison.py --config configs/polling_augmented_comparison_smoke.yaml
rtk python scripts/diagnose_polling_policy_shape.py --summary results/polling_augmented_smoke_comparison/20260428T155107Z/summary.json --json-output results/polling_augmented_policy_shape_diagnostic.json --markdown-output results/polling_augmented_policy_shape_diagnostic.md
```

Relevant tests:

```bash
rtk python -m unittest tests/test_polling.py tests/test_polling_features.py tests/test_polling_policy_inspection.py tests/test_polling_policy_shape_diagnostic.py tests/test_polling_comparison_runner.py tests/test_nnq.py tests/test_amq.py
```

## Caution

Do not expand polling to multiseed yet. The AMQ policy is still degenerate, so a
larger polling comparison would produce misleading performance tables.
