#!/usr/bin/env python3
"""Run routing AMQ for multiple seeds and aggregate debug diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.utils.config import load_config
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to multi-seed YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    seeds = tuple(int(seed) for seed in config["seeds"])
    base_config_path = Path(config["base_config"])
    base_config = load_config(base_config_path)
    _validate_base_config(base_config)

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "routing_amq_multiseed"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")

    seed_config_dir = run_dir / "seed_configs"
    seed_config_dir.mkdir()

    rows = []
    for seed in seeds:
        seed_config_path = seed_config_dir / f"seed_{seed}.yaml"
        seed_config = _seed_config(base_config, seed)
        seed_config_path.write_text(
            yaml.safe_dump(seed_config, sort_keys=False),
            encoding="utf-8",
        )
        row = _run_seed(seed, seed_config_path)
        rows.append(row)

    write_jsonl(run_dir / "seed_summaries.jsonl", rows)
    summary = {
        "benchmark": "routing",
        "algorithm": "amq",
        "base_config": str(base_config_path),
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")
    aggregate = summary["aggregate"]
    print(
        "summary: "
        f"seeds={list(seeds)} "
        f"average_cost_mean={aggregate['average_cost_mean']['mean']:.6g} "
        f"policy_gap_mean={aggregate['policy_gap_mean']['mean']:.6g} "
        f"over_defend_mean={aggregate['over_defend_states']['mean']:.6g} "
        f"q_gap_mean={aggregate['q_reference_gap_mean']['mean']:.6g}"
    )
    return 0


def _validate_base_config(config: dict[str, Any]) -> None:
    if config["env"]["name"] != "routing":
        raise ValueError("routing AMQ multi-seed runner requires env.name='routing'")
    if config["algorithm"]["name"] != "amq":
        raise ValueError("routing AMQ multi-seed runner requires algorithm.name='amq'")


def _seed_config(base_config: dict[str, Any], seed: int) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config.setdefault("amq", {})["seed"] = seed
    return config


def _run_seed(seed: int, config_path: Path) -> dict[str, Any]:
    start = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            str(config_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    runtime = perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"seed {seed} failed with code {completed.returncode}:\n{completed.stderr}"
        )
    child_run_dir = _run_dir_from_stdout(completed.stdout)
    summary = json.loads((child_run_dir / "summary.json").read_text(encoding="utf-8"))
    return _row(seed, config_path, child_run_dir, summary, runtime)


def _row(
    seed: int,
    config_path: Path,
    child_run_dir: Path,
    summary: dict[str, Any],
    runtime: float,
) -> dict[str, Any]:
    evaluation = summary.get("evaluation", {})
    policy_inspection = summary.get("policy_inspection", {})
    policy_comparison = summary.get("policy_comparison", {})
    q_diagnostic = summary.get("q_diagnostic", {})
    return {
        "seed": seed,
        "config": str(config_path),
        "run_dir": str(child_run_dir),
        "runtime_seconds": runtime,
        "average_cost_mean": evaluation.get("average_cost_mean"),
        "discounted_cost_mean": evaluation.get("discounted_cost_mean"),
        "defend_probability_mean": policy_inspection.get("defend_probability_mean"),
        "defend_states": policy_inspection.get("num_states_p_defend_at_least_threshold"),
        "policy_gap_mean": policy_comparison.get("p_defend_abs_gap_mean"),
        "over_defend_states": policy_comparison.get("num_states_amq_over_defends"),
        "under_defend_states": policy_comparison.get("num_states_amq_under_defends"),
        "amq_bellman_residual_mean": q_diagnostic.get(
            "amq_bellman_abs_residual_mean"
        ),
        "q_reference_gap_mean": q_diagnostic.get("q_reference_abs_gap_mean"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = (
        "average_cost_mean",
        "discounted_cost_mean",
        "defend_probability_mean",
        "defend_states",
        "policy_gap_mean",
        "over_defend_states",
        "under_defend_states",
        "amq_bellman_residual_mean",
        "q_reference_gap_mean",
        "runtime_seconds",
    )
    return {metric: _mean_std(rows, metric) for metric in metrics}


def _mean_std(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _run_dir_from_stdout(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("wrote "):
            return Path(line.removeprefix("wrote ").strip())
    raise RuntimeError(f"runner output did not contain run directory:\n{stdout}")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
