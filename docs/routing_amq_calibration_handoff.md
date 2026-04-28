# Routing AMQ Calibration Handoff

Date: 2026-04-28

## Current Goal

Build a reproducible, credible routing benchmark comparison for BVI / AMQ / NNQ.
The immediate AMQ goal is not to beat BVI by tuning noise, but to make the AMQ
policy structurally reasonable under the same routing environment, seeds, and
evaluation protocol.

## Implemented Since Last Clean Baseline

- Added routing three-algorithm comparison runner:
  - `scripts/run_routing_comparison.py`
  - `configs/routing_comparison_smoke.yaml`
  - `configs/routing_comparison_debug.yaml`
- Added routing multiseed comparison runner:
  - `scripts/run_routing_comparison_multiseed.py`
  - `configs/routing_comparison_multiseed_smoke.yaml`
  - `configs/routing_comparison_multiseed_debug.yaml`
- Added additional routing rollout evaluations:
  - `always_attack_evaluation`
  - algorithm-pair `minimax_evaluation`
  - common `bvi_attacker_evaluation`
- Added normalized routing AMQ feature:
  - `normalized_full_action_interaction` in `src/adversarial_queueing/features/routing_features.py`
  - `configs/routing_amq_normalized_debug.yaml`
  - `configs/routing_comparison_normalized_amq_debug.yaml`
  - `configs/routing_comparison_normalized_amq_multiseed_debug.yaml`
  - eval50 versions:
    - `configs/routing_comparison_normalized_amq_eval50_debug.yaml`
    - `configs/routing_comparison_normalized_amq_multiseed_eval50_debug.yaml`
- Added AMQ visitation diagnostics:
  - `rollout_state_visitation(...)` in `src/adversarial_queueing/evaluation/rollout.py`
  - `scripts/analyze_routing_amq_visitation.py`
  - `configs/routing_amq_normalized_visitation_seed0_debug.yaml`
- Added bounded fitted Bellman calibration for routing AMQ:
  - optional `AMQConfig` fields:
    - `fitted_calibration_passes`
    - `fitted_calibration_max_queue_length`
    - `fitted_calibration_eta`
  - config parsing in `src/adversarial_queueing/utils/config.py`
  - config:
    - `configs/routing_amq_normalized_fitted_debug.yaml`
  - comparison configs:
    - `configs/routing_comparison_normalized_fitted_amq_eval50_debug.yaml`
    - `configs/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug.yaml`
- Updated README Quick Start with routing comparison commands.

## Verification Already Run

```bash
rtk python -m unittest tests/test_routing_features.py tests/test_rollout_evaluation.py tests/test_routing_comparison_runner.py tests/test_routing_comparison_multiseed_runner.py
rtk python -m py_compile scripts/run_routing_comparison.py scripts/run_routing_comparison_multiseed.py scripts/run_experiment.py src/adversarial_queueing/features/routing_features.py src/adversarial_queueing/evaluation/rollout.py
rtk python -m py_compile scripts/analyze_routing_amq_visitation.py src/adversarial_queueing/evaluation/rollout.py
rtk python -m unittest tests/test_amq.py tests/test_routing_features.py tests/test_rollout_evaluation.py
rtk python -m py_compile src/adversarial_queueing/algorithms/amq.py src/adversarial_queueing/utils/config.py scripts/run_experiment.py scripts/analyze_routing_amq_visitation.py
```

All passed.

## Key Experiment Results

### Original routing comparison, 3 seeds, 10 eval episodes

Result file:

`results/routing_comparison_multiseed_debug/20260427T150515Z/summary.json`

Common BVI-attacker mean:

- BVI: `0.253411`
- NNQ: `0.260222`
- original AMQ: `0.283870`

Conclusion: original AMQ is materially worse than BVI/NNQ.

### Normalized AMQ, 3 seeds, 10 eval episodes

Result file:

`results/routing_comparison_normalized_amq_multiseed_debug/20260427T154312Z/summary.json`

Common BVI-attacker mean:

- BVI: `0.253411`
- normalized AMQ: `0.253762`
- NNQ: `0.260222`

Conclusion: normalized feature strongly improves AMQ and roughly matches BVI under 10-episode evaluation.

### Normalized AMQ, 3 seeds, 50 eval episodes

Result file:

`results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json`

Common BVI-attacker mean:

- BVI: `0.218785`
- normalized AMQ: `0.224463`
- NNQ: `0.228579`

Ranking counts under common BVI attacker:

- BVI: 2 seeds
- AMQ: 1 seed
- NNQ: 0 seeds

Conclusion: normalized AMQ is a real improvement and beats NNQ on mean, but it remains slightly weaker than BVI with lower-variance evaluation.

## Negative Calibration Attempts

Tried and deleted:

- normalized feature with lower constant eta (`0.0003`)
- normalized feature with Robbins-Monro schedule

These did not improve the seed 0 over-defense problem. Robbins-Monro made common BVI-attacker cost much worse (`0.2755` for seed 0).

Do not continue blind learning-rate tuning unless there is a new diagnostic reason.

## AMQ Failure Mode: Confirmed Over-Defense in Visited States

Result file:

`results/routing_amq_normalized_visitation_seed0_debug/20260427T170257Z/summary.json`

Important values:

- Visited states: `37 / 64`
- Visited over-defend states: `26`
- Visit-weighted policy gap: `0.597218`
- Visited abs gap mean: `0.723199`
- Unvisited abs gap mean: `0.974455`

High-visit examples:

- `[0,0,0]`: visited `727`, AMQ `p_defend=0.4897`, BVI `0.0`
- `[0,0,1]`: visited `221`, AMQ `1.0`, BVI `0.0`
- `[1,1,1]`: visited `167`, AMQ `1.0`, BVI `0.0`

Conclusion: over-defense is not an artifact of unvisited boundary states. It happens in core low-load states.

Manual Q-margin check from existing `q_diagnostic.jsonl` shows AMQ often estimates defender column 1 as lower cost than column 0 in low-load states where BVI does not. The failure is Q calibration, not just rollout variance.

## Bounded Fitted Bellman Calibration Result

Implemented the recommended bounded fitted Bellman calibration pass. This is an
AMQ-only self-calibration step after online learning; it enumerates routing
states within the configured queue bound and applies Bellman TD updates using
the environment transition model and AMQ's own continuation value. It does not
train on BVI values.

Config:

`configs/routing_amq_normalized_fitted_debug.yaml`

Parameters:

- normalized routing feature
- online AMQ steps: `100000`
- fitted passes: `5`
- fitted max queue length: `3`
- fitted eta: `0.0003`
- evaluation episodes: `50`

### Seed 0 AMQ-only run

Result file:

`results/routing_amq_normalized_fitted_debug/20260427T175859Z/summary.json`

Important values:

- random-attacker average cost: `0.200599`
- BVI-attacker average cost: `0.224124`
- policy gap mean: `0.371803`
- defend states: `30`
- AMQ over-defend states on full grid: `21`

Compared with normalized AMQ without fitted calibration, seed 0 policy sanity
improved substantially:

- over-defend states: `53 -> 21`
- policy gap mean: `0.829197 -> 0.371803`

The seed 0 BVI-attacker cost worsened slightly versus normalized AMQ without
fitted calibration (`0.219681 -> 0.224124`), but remained close enough to test
under multiseed evaluation.

### Seed 0 visitation diagnostic

Result file:

`results/routing_amq_normalized_fitted_debug/20260427T180135Z/summary.json`

Important values:

- visited states: `43 / 64`
- visited over-defend states: `13`
- visit-weighted policy gap: `0.397735`
- visited abs gap mean: `0.367334`
- unvisited abs gap mean: `0.380952`

This passed the previously defined go/no-go thresholds:

- visit-weighted gap below `0.40`
- visited over-defend states below `15`

### Fitted AMQ comparison, 3 seeds, 50 eval episodes

Result file:

`results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json`

Common BVI-attacker mean:

- BVI: `0.218785`
- fitted AMQ: `0.226511`
- NNQ: `0.228579`

Ranking counts under common BVI attacker:

- BVI: 3 seeds
- fitted AMQ: 0 seeds
- NNQ: 0 seeds

Per-seed BVI-attacker rankings:

- seed 0: BVI `0.220809`, fitted AMQ `0.224124`, NNQ `0.233069`
- seed 1: BVI `0.216846`, fitted AMQ `0.226492`, NNQ `0.226959`
- seed 2: BVI `0.218699`, fitted AMQ `0.228918`, NNQ `0.225708`

Policy sanity aggregate:

- fitted AMQ policy gap mean: `0.204343`
- fitted AMQ defend states: seed 0 `30`, seed 1 `18`, seed 2 `20`
- fitted AMQ BVI-attacker std: `0.001957`

Conclusion: fitted calibration is diagnostically useful because it reduces the
over-defense pathology sharply, especially on seed 0. It is not yet a clean
performance improvement: its 3-seed common BVI-attacker mean (`0.226511`) is
worse than normalized AMQ without fitted calibration (`0.224463`) by about
`0.00205`, just outside the prior acceptance margin. Keep it as an experimental
diagnostic branch/baseline, but do not present it as the main AMQ result.

## Report Generator Result

Next, do not continue blind eta/pass tuning. The evidence says fitted
calibration fixes much of the structural over-defense but gives back some
rollout cost. A comparison table generator has been added to collate existing
result summaries into machine-readable and human-readable artifacts. This helps
lock down the experiment narrative before more algorithm changes.

Implemented files:

- `scripts/build_routing_comparison_report.py`
- `tests/test_routing_comparison_report.py`
- `results/routing_comparison_report.json`
- `results/routing_comparison_report.md`

Rebuild command:

```bash
rtk python scripts/build_routing_comparison_report.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --normalized-visitation-summary results/routing_amq_normalized_visitation_seed0_debug/20260427T170257Z/summary.json --fitted-visitation-summary results/routing_amq_normalized_fitted_debug/20260427T180135Z/summary.json --json-output results/routing_comparison_report.json --markdown-output results/routing_comparison_report.md
```

Primary output file:

`results/routing_comparison_report.md`

Include at least:

- original AMQ eval50 if available, or mark not run
- normalized AMQ eval50:
  `results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json`
- fitted AMQ eval50:
  `results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json`
- seed 0 visitation before fitted:
  `results/routing_amq_normalized_visitation_seed0_debug/20260427T170257Z/summary.json`
- seed 0 visitation after fitted:
  `results/routing_amq_normalized_fitted_debug/20260427T180135Z/summary.json`

The report states the current best defensible AMQ baseline:

- main performance baseline: normalized AMQ without fitted calibration
- diagnostic/calibration baseline: normalized AMQ with fitted Bellman calibration

## Recommended Next Fully-Automatic Task

Add a small analysis script for where fitted AMQ loses rollout cost despite
lower policy gap. Start with existing per-seed summaries and avoid new long
runs unless the script identifies a concrete missing diagnostic.

This has now been implemented:

- `scripts/diagnose_routing_amq_cost_loss.py`
- `tests/test_routing_amq_cost_loss_diagnostic.py`
- generated machine-readable diagnostic:
  `results/routing_amq_cost_loss_diagnostic.json`
- generated human-readable diagnostic:
  `results/routing_amq_cost_loss_diagnostic.md`

Rebuild command:

```bash
rtk python scripts/diagnose_routing_amq_cost_loss.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --json-output results/routing_amq_cost_loss_diagnostic.json --markdown-output results/routing_amq_cost_loss_diagnostic.md
```

Key diagnostic result:

- fitted minus normalized BVI-attacker average cost: `+0.002048`
- fitted minus normalized random-attacker average cost: `-0.001146`
- fitted minus normalized always-attack average cost: `-0.000698`
- fitted minus normalized minimax average cost: `+0.003977`
- policy gap mean delta: `-0.151533`
- defend probability mean delta: `-0.187419`
- defend states delta: `-12.333333`

Interpretation: fitted calibration is not globally worse. It improves policy
shape and slightly improves random/always-attack costs, but it worsens all
three seeds under the common BVI attacker. The likely mechanism is that fitted
calibration removes broad over-defense, including some defense mass that was
useful specifically under BVI-attacker visitation.

Recommended next task:

Build a BVI-attacker-weighted visited-state policy-gap diagnostic. This should
compare normalized vs fitted AMQ on states actually visited under the common
BVI attacker, not on the full bounded grid. Do not tune eta/pass counts before
that diagnostic exists.

This has now been implemented and run:

- `scripts/diagnose_routing_bvi_attacker_visitation.py`
- `tests/test_routing_bvi_attacker_visitation_diagnostic.py`
- seed 0 smoke diagnostic:
  `results/routing_bvi_attacker_visitation_seed0_diagnostic.md`
- 3-seed diagnostic:
  `results/routing_bvi_attacker_visitation_diagnostic.md`
  `results/routing_bvi_attacker_visitation_diagnostic.json`

Rebuild command:

```bash
rtk python scripts/diagnose_routing_bvi_attacker_visitation.py --normalized-summary results/routing_comparison_normalized_amq_multiseed_eval50_debug/20260427T162942Z/summary.json --fitted-summary results/routing_comparison_normalized_fitted_amq_multiseed_eval50_debug/20260427T180933Z/summary.json --json-output results/routing_bvi_attacker_visitation_diagnostic.json --markdown-output results/routing_bvi_attacker_visitation_diagnostic.md
```

Key result, fitted minus normalized under BVI-attacker visitation:

- visit-weighted abs gap: `-0.073515`
- visit-weighted signed gap: `-0.091726`
- visited AMQ defend probability: `-0.082716`
- visited BVI defend probability: `+0.009010`
- visited over-defend states: `-5.000000`

Per seed, fitted AMQ lowers visited AMQ defend probability:

- seed 0: `-0.201222`
- seed 1: `-0.033506`
- seed 2: `-0.013419`

Interpretation: fitted calibration lowers AMQ defense probability on states
visited by the common BVI attacker. This explains the cost tradeoff: fitted
calibration removes global over-defense and improves policy-shape diagnostics,
but it also removes defense mass that was protecting cost-relevant
BVI-attacker trajectories.

Recommended next task:

Do not tune fitted eta/pass counts yet. First add a small report section that
ties together:

- main performance table
- cost-loss diagnostic
- BVI-attacker visitation diagnostic

This should make the routing result narrative explicit: normalized AMQ remains
the selected performance baseline; fitted AMQ is a calibration diagnostic that
reveals the defense-mass tradeoff.

This has now been implemented:

- `scripts/build_routing_narrative_report.py`
- `tests/test_routing_narrative_report.py`
- generated machine-readable narrative:
  `results/routing_narrative_report.json`
- generated human-readable narrative:
  `results/routing_narrative_report.md`

Rebuild command:

```bash
rtk python scripts/build_routing_narrative_report.py --comparison-report results/routing_comparison_report.json --cost-loss-diagnostic results/routing_amq_cost_loss_diagnostic.json --bvi-attacker-visitation-diagnostic results/routing_bvi_attacker_visitation_diagnostic.json --json-output results/routing_narrative_report.json --markdown-output results/routing_narrative_report.md
```

Narrative claim:

Normalized AMQ is the selected routing AMQ performance baseline. Fitted AMQ is
retained as a calibration diagnostic: it improves policy-shape diagnostics and
random/always-attack costs, but it reduces defense probability on states
visited by the common BVI attacker, which explains its worse BVI-attacker
rollout cost.

Recommended next task:

If continuing algorithm work, design a calibration variant that preserves
BVI-attacker-visited defense mass while reducing unvisited or low-value
over-defense. Do this only after defining an explicit acceptance criterion
against `results/routing_narrative_report.md`.

## Success Criteria For Next Agent

Prefer common BVI-attacker evaluation as the main cross-method metric.

The current best result for final narrative is normalized AMQ without fitted
calibration:

- BVI-attacker mean: `0.224463`
- beats NNQ mean: `0.228579`
- remains behind BVI mean: `0.218785`

The fitted-calibrated AMQ variant should be described as:

- useful diagnostic improvement in policy sanity
- not the selected performance baseline
- possible future work if a calibration objective can preserve rollout cost

## Files To Read First

- `CODEX.md`
- `experiment_spec_codex_draft.md`
- `src/adversarial_queueing/algorithms/amq.py`
- `src/adversarial_queueing/features/routing_features.py`
- `scripts/analyze_routing_amq_visitation.py`
- `scripts/run_routing_comparison_multiseed.py`
- this handoff file
