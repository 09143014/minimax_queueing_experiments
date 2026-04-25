# CODEX.md

Codex-facing engineering guidelines for the Approximate Minimax Q simulation project.

This file defines how Codex should work in this repository. It is intentionally strict: the project is a research simulation codebase, so correctness, reproducibility, and small verifiable changes matter more than fast-looking progress.

Tradeoff: these rules bias toward caution over speed. For trivial one-line edits, use judgment, but do not bypass reproducibility, tests, or experiment-output rules.

---

## 0. Project Context

This repository implements simulation experiments comparing three methods on adversarial queueing / control benchmarks:

- **NNQ**: neural-network Q baseline.
- **AMQ**: Approximate Minimax Q reinforcement-learning algorithm.
- **BVI**: bounded value-iteration / dynamic-programming-like algorithm.

The benchmark suite initially includes:

- routing system,
- polling system,
- service-rate-control system.

The goal is not merely to make code run. The goal is to produce credible, inspectable, reproducible experimental evidence comparing NNQ, AMQ, and BVI.

Important research framing:

- **BVI must not be hard-coded as ground truth.**
- NNQ and BVI should both be treated as baselines/references.
- Later paper framing may use whichever method empirically performs better as the stronger reference.
- Do not encode conclusions into the implementation.

---

## 1. Think Before Coding

Do not silently guess. Do not hide uncertainty. Surface tradeoffs before changing code.

Before implementing a non-trivial change, state:

1. What you believe the task is.
2. Which files/modules are likely affected.
3. What assumptions you are making.
4. How success will be verified.

Ask for clarification when ambiguity would materially affect design, experiment results, or paper claims.

Examples of ambiguity that must be surfaced:

- whether a cost is per-step, discounted cumulative, average-cost, or finite-horizon total cost;
- whether the attacker/defender act simultaneously or sequentially;
- whether a probability transition belongs in the environment, the algorithm, or the experiment config;
- whether a comparison metric should be mean, confidence interval, regret, final value, wall-clock time, or sample efficiency.

Do not proceed by inventing missing theory. If the mathematical model is underspecified, create the smallest explicit placeholder and mark it clearly with `TODO(model-spec)` rather than burying an assumption in code.

---

## 2. Simplicity First

Implement the minimum code that solves the requested problem. Nothing speculative.

Rules:

- No features beyond what was asked.
- No abstractions for a single use case.
- No large frameworks unless the repository already uses them.
- No generic plugin architecture unless multiple concrete users already exist.
- No premature distributed training, async orchestration, dashboarding, database storage, or experiment tracking service.
- If 200 lines could be 50 without losing clarity, rewrite it.

Prefer:

- plain Python modules over clever metaprogramming;
- small dataclasses / typed configs over global mutable state;
- explicit loops over opaque abstractions when the loop is part of the research logic;
- readable numerical code over maximally clever vectorization, unless performance is already a measured bottleneck.

The test: would a senior research engineer say this is over-engineered for the current experiment? If yes, simplify.

---

## 3. Surgical Changes

Touch only what the task requires. Clean up only the mess your change creates.

When editing existing code:

- Do not reformat unrelated files.
- Do not rename public objects unless required.
- Do not change comments you do not fully understand.
- Do not refactor adjacent code just because it looks imperfect.
- Match the existing style unless it is directly harmful.
- If you notice unrelated dead code or bugs, mention them separately; do not fix them opportunistically.

When your change creates unused imports, variables, functions, configs, or tests, remove those specific orphans.

Every changed line should trace directly to the user request or a verification requirement.

---

## 4. Goal-Driven Execution

Convert tasks into verifiable goals. Loop until the goal is met or a real blocker is found.

For multi-step work, use this pattern:

```text
1. Implement [specific change] -> verify with [specific test/check]
2. Update [specific experiment/config] -> verify with [small run/smoke test]
3. Save/report [specific artifact] -> verify with [file existence/schema/plot check]
```

Examples:

- “Add service-rate-control benchmark” means: define environment dynamics, add tests for transitions/costs, add minimal config, run a smoke experiment.
- “Fix AMQ bug” means: create or identify a failing test or reproducible run, fix the bug, prove the test/run now passes.
- “Compare NNQ and BVI” means: ensure both use the same benchmark config, seeds, budget conventions, and metrics before reporting results.

Do not claim success without verification.

---

## 5. Research Correctness Rules

### 5.1 Separate environment, algorithm, and experiment code

Keep these concerns separate:

- **Environment / benchmark**: state, action, transition, reward/cost, terminal/horizon logic.
- **Algorithm**: AMQ, NNQ, BVI update logic and policy/value estimation.
- **Experiment runner**: seeds, budgets, configs, repeated trials, logging, aggregation.
- **Analysis**: tables, plots, statistical summaries.

Do not let algorithm code contain benchmark-specific hacks.

Do not let benchmark code know which algorithm is being trained, except through a clean interface.

### 5.2 No hidden ground truth leakage

Do not use BVI outputs to train, tune, initialize, normalize, early-stop, or shape AMQ/NNQ unless the experiment explicitly says that is the condition being tested.

Do not let NNQ or AMQ read exact dynamic-programming values except in clearly named evaluation-only scripts.

Any oracle-like information must be isolated behind an explicit `evaluation` or `analysis` path.

### 5.3 Comparable budgets

When comparing methods, make budgets explicit:

- number of environment steps;
- number of episodes;
- number of Bellman/backups/iterations;
- wall-clock time if relevant;
- number of random seeds;
- training/evaluation split.

Do not compare methods using hidden unequal budgets.

### 5.4 Determinism and reproducibility

Every experiment must be reproducible from:

- code commit;
- config file;
- random seed list;
- command line;
- output directory.

Use explicit seeds for Python, NumPy, PyTorch/JAX if present, and environment RNGs.

Do not rely on global RNG state when a local RNG can be passed explicitly.

### 5.5 Cost/reward sign conventions

This project is primarily cost/minimax oriented. Be explicit about whether a function returns:

- cost to minimize;
- reward to maximize;
- value under defender objective;
- value under attacker objective.

Never silently flip signs. If an algorithm expects rewards but the benchmark defines costs, implement a clearly named adapter and test it.

---

## 6. Benchmark-Specific Rules

### 6.1 Routing system

Keep routing dynamics and routing topology/config separate from AMQ/NNQ/BVI.

Tests should cover at least:

- valid state/action shapes;
- transition conservation or queue-update invariants;
- cost monotonicity with congestion when applicable;
- deterministic behavior under fixed seeds.

### 6.2 Polling system

Keep polling order, service discipline, arrival process, and cost definition explicit in config.

Tests should cover at least:

- valid polling actions;
- queue update logic;
- boundary conditions for empty queues;
- deterministic behavior under fixed seeds.

### 6.3 Service-rate-control system

The initial implementation must follow these modeling choices:

- Service rates are initially **three discrete levels**.
- Congestion cost is **quadratic**.
- If the attacker attacks and the defender does not defend, the realized service rate is forced to the **lowest service-rate level**.

Implementation rules:

- Do not encode the lowest service-rate attack effect in AMQ, NNQ, or BVI. It belongs in the environment transition/dynamics.
- Keep nominal service-rate levels configurable, but default to three levels.
- Keep the attack/defense interaction explicit and tested.
- Test the exact case: attacker attacks + defender does not defend -> realized service rate equals the lowest level.
- Test the quadratic congestion cost separately from transition logic.

---

## 7. Algorithm Rules

### 7.1 AMQ

AMQ should be implemented as the project’s reinforcement-learning method for minimax/adversarial control.

Rules:

- Keep minimax target computation explicit and unit-tested where possible.
- Do not mix environment-specific assumptions into update rules.
- Log training curves needed to diagnose instability.
- Make exploration schedule, learning rate, discount/horizon, replay/batch settings, and network architecture explicit in config if applicable.

### 7.2 NNQ

NNQ is the neural-network Q baseline.

Rules:

- Keep NNQ baseline simple and credible.
- Do not intentionally weaken NNQ to make AMQ look better.
- Use the same evaluation protocol as AMQ unless the experiment explicitly justifies a difference.
- If NNQ cannot solve a minimax objective directly, document the adapter/objective clearly.

### 7.3 BVI

BVI is a bounded / value-iteration-like algorithm.

Rules:

- BVI is not ground truth by default.
- Do not hard-code BVI as the “correct answer” in tests, plots, or result labels.
- It is acceptable to test internal Bellman consistency on tiny hand-checkable MDPs.
- In experiment outputs, label it as `BVI`, `BVI baseline`, or `BVI reference`, not `ground_truth` unless a specific tiny analytic test actually proves that.

---

## 8. Interfaces and Code Organization

Prefer a simple structure like this unless the repository already has a better one:

```text
src/
  amq/
    algorithms/
      amq.py
      nnq.py
      bvi.py
    benchmarks/
      routing.py
      polling.py
      service_rate_control.py
    experiments/
      run_experiment.py
      configs.py
      evaluate.py
    analysis/
      aggregate.py
      plot.py
    utils/
      rng.py
      logging.py
      typing.py
tests/
  test_benchmarks/
  test_algorithms/
  test_experiments/
configs/
  routing.yaml
  polling.yaml
  service_rate_control.yaml
outputs/
  .gitkeep
```

This structure is a suggestion, not permission to rewrite the repo. If existing structure differs, follow the existing structure and preserve local conventions.

Core environment interface should be small and explicit. A benchmark should expose only what algorithms need, such as:

- reset / initial state;
- valid actions;
- transition or step;
- cost/reward;
- horizon/discount metadata;
- optional exact model access for algorithms that legitimately require it, such as BVI.

Do not create a broad generic interface until at least two benchmarks truly need the same abstraction.

---

## 9. Testing Standards

Add or update tests for every non-trivial change.

Minimum test categories:

- **Unit tests** for benchmark dynamics and cost functions.
- **Unit tests** for algorithm update formulas on tiny cases.
- **Smoke tests** for each benchmark-method pair where feasible.
- **Regression tests** for fixed-seed behavior when results should be deterministic.
- **Config tests** ensuring required experiment fields are present.

Tests should be small and fast by default. Long experiments should not run in ordinary unit tests.

Use tiny deterministic environments for algorithm correctness. Do not use expensive simulations to test basic formulas.

Do not assert fragile floating-point equality. Use tolerances and explain them.

---

## 10. Experiment Output Standards

Every experiment run should write a self-contained output directory containing:

```text
outputs/<experiment_name>/<timestamp_or_run_id>/
  config.yaml or config.json
  command.txt
  git_commit.txt          # if available
  metrics.csv or metrics.jsonl
  summary.json
  logs.txt
  plots/                  # optional
  checkpoints/            # optional
```

Rules:

- Do not overwrite previous results unless explicitly requested.
- Include seed in every per-run metric row.
- Include benchmark name, method name, and config hash or config path in outputs.
- Keep raw metrics separate from aggregated summaries.
- Make plots reproducible from saved raw metrics.

---

## 11. Metrics and Reporting

Report enough information to compare methods honestly:

- mean and variability across seeds;
- final performance and learning curve when applicable;
- sample budget and/or iteration budget;
- evaluation protocol;
- failure/instability cases.

Do not cherry-pick seeds.

Do not silently drop failed runs. Mark them and explain why.

Do not use plot titles, filenames, or variable names that imply a conclusion not supported by the run.

Bad:

```text
amq_beats_bvi.png
bvi_ground_truth_error.csv
```

Better:

```text
method_comparison_service_rate_control.png
value_estimate_gap_to_bvi_reference.csv
```

---

## 12. Configuration Rules

Prefer explicit configs over hidden constants.

Config should include, when relevant:

- benchmark name and parameters;
- method name and parameters;
- horizon / discount / stopping criteria;
- training budget;
- evaluation budget;
- seeds;
- output path;
- logging level;
- device selection if neural networks are used.

Do not bury experimental choices in scripts.

Do not duplicate the same parameter across multiple files unless there is a clear precedence rule.

When adding a config field, document its meaning and default.

---

## 13. Numerical and ML Engineering Rules

- Be explicit about tensor shapes.
- Check dimensions at module boundaries.
- Avoid silent broadcasting when it could hide a bug.
- Keep dtype/device conversions localized and visible.
- Clip gradients only if configured and logged.
- Do not normalize targets, costs, or rewards without documenting the exact transformation.
- Avoid nondeterministic GPU assumptions in tests.
- Keep training loops boring and inspectable.

For neural code, log enough diagnostics to debug:

- loss;
- value estimates;
- evaluation cost/reward;
- exploration parameter if any;
- gradient norm if useful;
- number of environment steps/backups.

---

## 14. Dependency Rules

Do not add a new dependency unless:

1. it clearly reduces code complexity or improves reliability;
2. it is actively maintained;
3. the same result would be meaningfully worse without it;
4. the dependency is added to the appropriate project file;
5. tests or smoke checks still pass.

Prefer standard library, NumPy, and the repository’s existing ML stack.

Do not introduce heavyweight experiment platforms, web servers, databases, or dashboards unless explicitly requested.

---

## 15. Documentation Rules

Document research logic, not obvious syntax.

Good comments explain:

- modeling choices;
- sign conventions;
- minimax objective conventions;
- why an approximation is acceptable;
- why a test case is theoretically meaningful.

Bad comments repeat code:

```python
x += 1  # increment x
```

Every benchmark should have a short docstring or README section explaining:

- state space;
- action space;
- attacker action, if any;
- defender action, if any;
- transition logic;
- cost function;
- horizon/discount convention.

---

## 16. What Not To Do

Do not:

- hard-code expected paper conclusions;
- make AMQ look better by weakening NNQ or BVI;
- use BVI as hidden supervision unless explicitly requested;
- add broad abstractions before there are multiple real users;
- rewrite the whole repository for style;
- change experiment semantics while “cleaning up” code;
- silently change default seeds or budgets;
- overwrite old outputs;
- hide failed runs;
- report plots without the raw metrics needed to regenerate them;
- claim convergence without a defined convergence criterion;
- claim optimality unless it is proven or verified for a tiny exact case.

---

## 17. Completion Checklist

Before saying a task is done, verify the relevant items:

- [ ] The implementation matches the requested model/algorithm change.
- [ ] Assumptions are explicit.
- [ ] No unrelated files were changed.
- [ ] Tests were added or updated for non-trivial logic.
- [ ] Unit tests or smoke tests were run, or the reason they could not be run is stated.
- [ ] Configs contain new parameters with documented defaults.
- [ ] Experiment outputs are reproducible and do not overwrite prior runs.
- [ ] Method labels do not encode unsupported conclusions.
- [ ] BVI is not treated as ground truth except in explicitly justified tiny exact tests.
- [ ] Service-rate-control rules are preserved: three initial service levels, quadratic congestion cost, and attack-without-defense forces lowest service rate.

---

## 18. How To Report Back

When reporting progress or final results, use this format:

```text
Summary:
- What changed.

Verification:
- Tests/checks run and outcomes.

Assumptions:
- Any assumptions made.

Notes:
- Any follow-up issues noticed but not changed.
```

Do not bury failures. If verification failed, say exactly what failed and what remains uncertain.
