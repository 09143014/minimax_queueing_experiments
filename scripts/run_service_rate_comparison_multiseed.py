#!/usr/bin/env python3
"""Run service-rate-control comparison over multiple seeds."""

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


METHODS = ("bvi", "amq", "nnq")
METRICS = (
    "average_cost_mean",
    "discounted_cost_mean",
    "boundary_hit_fraction_mean",
    "tail_fraction_mean",
    "first_state_p_high_at_least_threshold",
    "first_state_p_medium_at_least_threshold",
    "runtime_seconds",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to multiseed YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    seeds = tuple(int(seed) for seed in config["seeds"])
    base_config_path = Path(config["base_config"])
    base_config = load_config(base_config_path)
    _validate_base_config(base_config)

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "service_rate_comparison_multiseed"),
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
        print(f"running seed {seed}", flush=True)
        comparison_config_path = _write_seed_config(
            base_config,
            seed,
            seed_config_dir / f"seed_{seed}",
        )
        row = _run_seed(seed, comparison_config_path)
        rows.append(row)
        best = row["comparison"]["best_average_cost_methods"]
        print(f"finished seed {seed}: best_average_cost={best}", flush=True)

    write_jsonl(run_dir / "seed_method_summaries.jsonl", rows)
    summary = {
        "benchmark": "service_rate_control",
        "base_config": str(base_config_path),
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "ranking_counts": _ranking_counts(rows),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")
    aggregate = summary["aggregate"]
    print(
        "summary: "
        f"seeds={list(seeds)} "
        f"best_counts={summary['ranking_counts']['average_cost']} "
        f"amq_mean={aggregate['amq']['average_cost_mean']['mean']:.6g} "
        f"nnq_mean={aggregate['nnq']['average_cost_mean']['mean']:.6g} "
        f"bvi_mean={aggregate['bvi']['average_cost_mean']['mean']:.6g}"
    )
    return 0


def _validate_base_config(config: dict[str, Any]) -> None:
    if "runs" not in config:
        raise ValueError("multiseed base_config must be a service-rate comparison config")
    methods = [str(run["method"]) for run in config["runs"]]
    for method in METHODS:
        if method not in methods:
            raise ValueError(f"base_config is missing method {method}")


def _write_seed_config(
    base_config: dict[str, Any],
    seed: int,
    seed_dir: Path,
) -> Path:
    seed_dir.mkdir()
    runs = []
    for run in base_config["runs"]:
        method = str(run["method"])
        source_path = Path(run["config"])
        method_config = load_config(source_path)
        if method in {"amq", "nnq"}:
            method_config.setdefault(method, {})["seed"] = seed
        evaluation = method_config.setdefault("evaluation", {})
        evaluation["seed"] = int(evaluation.get("seed", 0)) + seed
        method_config_path = seed_dir / f"{method}.yaml"
        method_config_path.write_text(
            yaml.safe_dump(method_config, sort_keys=False),
            encoding="utf-8",
        )
        runs.append({"method": method, "config": str(method_config_path)})

    comparison_config = json.loads(json.dumps(base_config))
    comparison_config["runs"] = runs
    comparison_config.setdefault("experiment", {})["name"] = (
        f"service_rate_comparison_seed_{seed}"
    )
    comparison_config["experiment"]["output_dir"] = str(seed_dir / "children")

    comparison_config_path = seed_dir / "comparison.yaml"
    comparison_config_path.write_text(
        yaml.safe_dump(comparison_config, sort_keys=False),
        encoding="utf-8",
    )
    return comparison_config_path


def _run_seed(seed: int, config_path: Path) -> dict[str, Any]:
    start = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_service_rate_comparison.py"),
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
    return {
        "seed": seed,
        "config": str(config_path),
        "run_dir": str(child_run_dir),
        "runtime_seconds": runtime,
        "comparison": _comparison(summary["rows"]),
        "method_rows": summary["rows"],
    }


def _comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row["average_cost_mean"]))
    best_cost = float(ranked[0]["average_cost_mean"])
    return {
        "ranked_by_average_cost": [row["method"] for row in ranked],
        "best_average_cost_methods": [
            row["method"]
            for row in ranked
            if float(row["average_cost_mean"]) == best_cost
        ],
        "average_cost_gap_from_best": {
            row["method"]: float(row["average_cost_mean"]) - best_cost
            for row in rows
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregate = {}
    for method in METHODS:
        method_rows = [_method_row(row, method) for row in rows]
        aggregate[method] = {
            metric: _mean_std(method_rows, metric)
            for metric in METRICS
            if all(item.get(metric) is not None for item in method_rows)
        }
    return aggregate


def _method_row(row: dict[str, Any], method: str) -> dict[str, Any]:
    for method_row in row["method_rows"]:
        if method_row["method"] == method:
            return method_row
    raise ValueError(f"seed {row['seed']} is missing method {method}")


def _ranking_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {"average_cost": {method: 0 for method in METHODS}}
    for row in rows:
        for method in row["comparison"]["best_average_cost_methods"]:
            counts["average_cost"][method] += 1
    return counts


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
