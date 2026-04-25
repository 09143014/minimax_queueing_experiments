# Repository Operations Guide for Codex

This document defines how Codex should operate on this repository. It is a repository-management and GitHub-workflow guide, not an algorithm-design document.

The project is a research simulation codebase for comparing **NNQ**, **AMQ**, and **BVI** on multiple benchmark environments. Because experimental validity and reproducibility matter, repository operations must be conservative, traceable, and easy to review.

---

## 1. Core Principle

Codex should use the following workflow:

> **Create a minimal runnable prototype first; after that, make small, reviewable changes.**

Do not choose between “push every tiny edit” and “build a huge project privately.” The required policy is:

1. First establish a clean, minimal, runnable repository skeleton.
2. Push that skeleton to GitHub as the initial baseline.
3. After the baseline exists, make changes in small, focused branches and pull requests.

---

## 2. Initial Repository Baseline

Before the repository is treated as active, Codex should help create a minimal runnable prototype.

The initial baseline should include at least:

```text
README.md
CODEX.md
REPOSITORY_OPERATIONS.md
pyproject.toml or requirements.txt
src/
  amq/
  nnq/
  bvi/
  envs/
  experiments/
configs/
  smoke.yaml
tests/
scripts/
  run_experiment.py
```

The baseline does not need to contain final algorithms or full experiment results. It must, however, satisfy these conditions:

```bash
python scripts/run_experiment.py --config configs/smoke.yaml
pytest
```

If these commands are not yet possible, Codex should clearly state what is missing and implement only the minimum necessary scaffolding to make them possible.

---

## 3. Public vs Private Repository Policy

### Private repository

If the repository is private, Codex may push the initial skeleton early, even if some modules are placeholders.

The goal is version control, backup, and early structure.

### Public repository

If the repository is public, Codex must not push a messy or unexplained dump.

Before public release, the repository should contain:

- a readable `README.md`;
- a clear `CODEX.md` or `AGENTS.md` instruction file;
- a runnable smoke test;
- no secrets, tokens, local paths, or personal files;
- no large untracked experiment artifacts;
- no misleading claim that results are final.

---

## 4. Branch Policy

Codex must not work directly on `main` except for documentation-only emergency fixes explicitly requested by the human maintainer.

The preferred branch model is:

```text
main          stable runnable version only
dev           optional integration branch
feature/*     one task per Codex branch
fix/*         one bug fix per branch
docs/*        documentation-only changes
experiment/*  experimental changes that may not be merged
```

Recommended branch examples:

```text
feature/init-project-skeleton
feature/add-routing-env
feature/add-polling-env
feature/add-service-rate-control-env
feature/add-nnq-baseline
feature/add-amq-training-loop
feature/add-bvi-baseline
feature/add-smoke-tests
fix/service-rate-transition-bug
docs/update-reproducibility-guide
```

Each branch should have one clear purpose.

---

## 5. Pull Request Policy

Codex should push changes to a feature branch and open a pull request instead of pushing directly to `main`.

A pull request should include:

1. What changed.
2. Why it changed.
3. How it was tested.
4. Any known limitations.
5. Whether experiment outputs are affected.

Suggested PR description format:

```markdown
## Summary
- 

## Motivation
- 

## Changes
- 

## Tests
- [ ] `pytest`
- [ ] `python scripts/run_experiment.py --config configs/smoke.yaml`

## Reproducibility impact
- Seeds affected: yes/no
- Configs affected: yes/no
- Existing results invalidated: yes/no/unknown

## Notes for reviewer
- 
```

Codex should assume the human maintainer reviews and merges PRs.

---

## 6. Commit Granularity

After the initial baseline exists, Codex must use small, meaningful commits.

Good commit examples:

```text
Add routing environment transition logic
Add polling environment smoke test
Implement NNQ replay buffer
Add AMQ attacker-defender update loop
Add BVI finite-horizon solver skeleton
Add service-rate-control quadratic congestion cost
Fix seed handling in experiment runner
Document output directory format
```

Bad commit examples:

```text
update
fix
misc
changes
final
try this
big refactor
```

A commit should usually change one conceptual unit:

- one environment;
- one algorithm component;
- one metric;
- one config format;
- one test group;
- one documentation topic.

Do not mix environment logic, algorithm changes, plotting, and documentation in the same commit unless the task explicitly requires it.

---

## 7. Codex GitHub Permissions and Safety Rules

Codex may:

- create local commits;
- create feature branches;
- push feature branches;
- open pull requests;
- update pull request branches;
- respond to review comments with additional commits.

Codex must not:

- push directly to `main`;
- force-push shared branches unless explicitly authorized;
- delete branches without permission;
- rewrite public history;
- commit secrets, API keys, tokens, credentials, or local machine paths;
- commit large raw experiment outputs unless explicitly requested;
- silently change benchmark definitions;
- silently change random seed handling;
- silently change evaluation metrics.

If a requested task appears to require one of the forbidden actions, Codex should stop and explain the risk.

---

## 8. Required Pre-Commit Checks

Before committing, Codex should run the smallest relevant checks.

For documentation-only changes:

```bash
git diff --check
```

For code changes:

```bash
pytest
python scripts/run_experiment.py --config configs/smoke.yaml
```

For changes affecting formatting or static checks, run the project-defined commands, for example:

```bash
ruff check .
ruff format --check .
mypy src
```

Only run tools that are actually configured in the repository. Do not invent unavailable tooling.

If a check fails, Codex should either fix the issue or clearly report the failure in the PR description.

---

## 9. Experimental Reproducibility Rules

Because this repository supports research experiments, Codex must treat reproducibility as part of the Git workflow.

Any change to the following must be isolated and clearly documented:

- random seed logic;
- environment transition rules;
- reward or cost definitions;
- attacker or defender action spaces;
- NNQ training loop;
- AMQ update rule;
- BVI solver logic;
- evaluation metrics;
- benchmark configs;
- plotting scripts;
- result serialization.

When such a change is made, the PR must state whether existing experiment outputs may be invalidated.

Do not modify benchmark semantics as part of a cleanup or refactor commit.

---

## 10. Handling Results and Artifacts

Codex should not commit large generated artifacts by default.

Do not commit by default:

```text
outputs/
results/
runs/
wandb/
mlruns/
*.pkl
*.pt
*.pth
*.npy
*.npz
*.csv    # unless small and intentionally versioned
```

Small smoke-test fixtures may be committed if they are required for tests.

Large or final paper-level results should be stored according to a separate artifact policy, not casually added to Git.

If generated outputs are necessary to demonstrate a change, Codex should summarize them in the PR description instead of committing them.

---

## 11. `.gitignore` Expectations

The repository should ignore common local and generated files, including:

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
.env
.env.*
.DS_Store
outputs/
results/
runs/
wandb/
mlruns/
related papers/
```

Codex should update `.gitignore` when a new generated directory or cache appears.

Do not use `.gitignore` to hide source files that should be reviewed.

---

## 12. Review Workflow with Codex

Codex may be asked to review pull requests.

When reviewing, Codex should prioritize:

1. benchmark semantic changes;
2. reproducibility risks;
3. seed handling;
4. incorrect algorithm comparisons;
5. tests that do not actually exercise the changed behavior;
6. hidden dependency or config assumptions;
7. accidental large artifacts;
8. unclear documentation.

For this project, review comments should treat the following as high priority:

- BVI being treated as hard-coded ground truth;
- NNQ, AMQ, and BVI using inconsistent evaluation protocols;
- service-rate-control environment violating the defined attack rule;
- service-rate-control congestion cost not being quadratic;
- service-rate-control initial service rates not using the agreed discrete levels;
- experiments that cannot be reproduced from config and seed.

---

## 13. Merge Policy

A pull request may be merged only when:

- the diff is understandable;
- tests or smoke runs pass, or failures are explicitly justified;
- no secrets or large accidental artifacts are included;
- the PR has a clear purpose;
- benchmark semantics are documented if changed;
- the human maintainer approves.

Codex should not self-merge unless explicitly instructed.

---

## 14. Rollback Policy

If a Codex change causes breakage, prefer a clean revert over ad hoc patching.

Recommended rollback flow:

```bash
git revert <commit>
```

For unmerged PR branches, Codex may add a corrective commit instead of rewriting history.

Avoid force-pushing unless the branch is private to the current task and the human maintainer has approved it.

---

## 15. Task Completion Checklist for Codex

Before reporting a task as complete, Codex should answer:

- Did I work on a feature/fix/docs branch rather than `main`?
- Is the diff focused?
- Did I avoid unrelated refactors?
- Did I run the relevant checks?
- Did I avoid committing generated outputs or secrets?
- Did I document reproducibility impact?
- Did I preserve benchmark semantics unless explicitly asked to change them?
- Did I open or update a PR instead of pushing directly to `main`?

If any answer is “no,” Codex should disclose it clearly.

---

## 16. Default Operating Rule

When uncertain, Codex should choose the safer repository action:

```text
small branch > direct main edit
pull request > direct push
smoke test > unverified change
clear note > silent assumption
revert > messy repair
```

The goal is not to maximize code volume. The goal is to keep the research codebase understandable, reproducible, and reviewable.
