#!/usr/bin/env python3
"""Run routing BVI/AMQ/NNQ comparison over multiple seeds."""

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


ALGORITHMS = ("bvi", "amq", "nnq")
METRICS = (
    "average_cost_mean",
    "always_attack_average_cost_mean",
    "minimax_average_cost_mean",
    "bvi_attacker_average_cost_mean",
    "policy_gap_mean",
    "q_reference_gap_mean",
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
        config["experiment"].get("name", "routing_comparison_multiseed"),
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
        best = row["comparison"]["bvi_attacker_best_average_cost_algorithms"]
        print(f"finished seed {seed}: bvi_attacker_best={best}", flush=True)

    write_jsonl(run_dir / "seed_algorithm_summaries.jsonl", rows)
    summary = {
        "benchmark": "routing",
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
        f"bvi_attacker_best_counts={summary['ranking_counts']['bvi_attacker']} "
        f"bvi_attacker_amq_mean={aggregate['amq']['bvi_attacker_average_cost_mean']['mean']:.6g} "
        f"bvi_attacker_nnq_mean={aggregate['nnq']['bvi_attacker_average_cost_mean']['mean']:.6g} "
        f"bvi_attacker_bvi_mean={aggregate['bvi']['bvi_attacker_average_cost_mean']['mean']:.6g}"
    )
    return 0


def _validate_base_config(config: dict[str, Any]) -> None:
    if "configs" not in config:
        raise ValueError("multiseed base_config must be a routing comparison config")
    for algorithm in ALGORITHMS:
        if algorithm not in config["configs"]:
            raise ValueError(f"base_config is missing configs.{algorithm}")


def _write_seed_config(
    base_config: dict[str, Any],
    seed: int,
    seed_dir: Path,
) -> Path:
    seed_dir.mkdir()
    algorithm_config_paths = {}
    for algorithm in ALGORITHMS:
        source_path = Path(base_config["configs"][algorithm])
        algorithm_config = load_config(source_path)
        if algorithm in {"amq", "nnq"}:
            algorithm_config.setdefault(algorithm, {})["seed"] = seed
        algorithm_config_path = seed_dir / f"{algorithm}.yaml"
        algorithm_config_path.write_text(
            yaml.safe_dump(algorithm_config, sort_keys=False),
            encoding="utf-8",
        )
        algorithm_config_paths[algorithm] = str(algorithm_config_path)

    comparison_config = json.loads(json.dumps(base_config))
    comparison_config["configs"] = algorithm_config_paths
    comparison_config.setdefault("experiment", {})["name"] = (
        f"routing_comparison_seed_{seed}"
    )
    comparison_config["experiment"]["output_dir"] = str(seed_dir / "children")
    shared = comparison_config.setdefault("shared", {})
    evaluation = shared.setdefault("evaluation", {})
    evaluation["seed"] = int(evaluation.get("seed", 0)) + seed

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
            str(ROOT / "scripts" / "run_routing_comparison.py"),
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
        "comparison": summary["comparison"],
        "algorithm_rows": summary["rows"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregate = {}
    for algorithm in ALGORITHMS:
        algorithm_rows = [
            _algorithm_row(row, algorithm)
            for row in rows
        ]
        aggregate[algorithm] = {
            metric: _mean_std(algorithm_rows, metric)
            for metric in METRICS
            if all(item.get(metric) is not None for item in algorithm_rows)
        }
    return aggregate


def _algorithm_row(row: dict[str, Any], algorithm: str) -> dict[str, Any]:
    for algorithm_row in row["algorithm_rows"]:
        if algorithm_row["algorithm"] == algorithm:
            return algorithm_row
    raise ValueError(f"seed {row['seed']} is missing algorithm {algorithm}")


def _ranking_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    rankings = {
        "random_attacker": "random_attacker_best_average_cost_algorithms",
        "always_attack": "always_attack_best_average_cost_algorithms",
        "minimax": "minimax_best_average_cost_algorithms",
        "bvi_attacker": "bvi_attacker_best_average_cost_algorithms",
    }
    counts = {
        ranking_name: {algorithm: 0 for algorithm in ALGORITHMS}
        for ranking_name in rankings
    }
    for row in rows:
        comparison = row["comparison"]
        for ranking_name, key in rankings.items():
            for algorithm in comparison[key]:
                counts[ranking_name][algorithm] += 1
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
