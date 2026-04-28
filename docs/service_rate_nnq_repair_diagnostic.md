# Service-Rate NNQ Repair Diagnostic

Date: 2026-04-28

## Goal

Test whether the service-rate NNQ failure diagnosed earlier is actually caused
by empty-state high-service overuse, rather than by a broader service-rate model
failure.

This is a repair diagnostic, not a new claimed NNQ training algorithm. The NNQ
network is trained unchanged. Only rollout-time defender policies are compared:

- baseline NNQ policy;
- fixed low or medium service for states `x <= k`, then NNQ elsewhere.

## Command

```bash
rtk python scripts/diagnose_service_rate_nnq_repair.py --config configs/nnq_low_state_repair_debug.yaml --json-output results/service_rate_nnq_repair_diagnostic.json --markdown-output results/service_rate_nnq_repair_diagnostic.md
```

## Result

Run directory:

`results/service_rate_control_nnq_low_state_repair_debug/20260428T122135Z`

Seeds: `0, 1, 2`

Average rollout cost:

- baseline NNQ: `0.517298`
- best repair, `state <= 0 -> low service`: `0.392079`
- improvement: `0.125219`

The best candidate was the same on all three seeds:

`state_le_0_action_0`

Candidate means:

| Candidate | Mean average cost | State 0 p_high |
|---|---:|---:|
| baseline | `0.517298` | `1.000` |
| state <= 0 -> low | `0.392079` | `0.000` |
| state <= 0 -> medium | `0.428676` | `0.000` |
| state <= 1 -> low | `0.519921` | `0.000` |
| state <= 1 -> medium | `0.486943` | `0.000` |
| state <= 2 -> low | `0.827857` | `0.000` |
| state <= 2 -> medium | `0.688479` | `0.000` |

## Interpretation

The repair diagnostic supports the earlier policy-shape diagnosis: NNQ's
service-rate-control cost is substantially driven by choosing high service at
the empty queue.

The result is also narrow. Repairing only `state == 0` helps. Extending the same
fixed action to `state <= 1` or `state <= 2` is neutral or harmful, so a broad
low-state override is not justified.

Current service-rate ranking remains:

- BVI mean from the consolidation run: `0.317933`
- AMQ mean from the consolidation run: `0.363171`
- repaired NNQ diagnostic mean: `0.392079`
- baseline NNQ mean: `0.517298`

This means the repair closes much of NNQ's gap but does not overtake AMQ or BVI
under the current debug protocol.

## Next Step

If service-rate NNQ needs to become a stronger baseline, the next minimal
algorithmic experiment should be a clean, documented `state == 0` correction
condition or an equivalent target calibration that specifically prevents
positive service cost at the empty queue from being undervalued. It should be
reported separately from raw NNQ because it injects benchmark-specific structure.
