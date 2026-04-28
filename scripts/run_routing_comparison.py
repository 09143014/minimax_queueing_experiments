#!/usr/bin/env python3
"""Run routing BVI, AMQ, and NNQ configs and aggregate comparable diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.utils.config import load_config
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


ALGORITHMS = ("bvi", "amq", "nnq")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to comparison YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config_paths = {
        algorithm: Path(config["configs"][algorithm]) for algorithm in ALGORITHMS
    }
    base_configs = {
        algorithm: _comparison_config(load_config(path), algorithm, config)
        for algorithm, path in config_paths.items()
    }
    _validate_base_configs(base_configs)

    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "routing_comparison"),
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")

    generated_config_dir = run_dir / "algorithm_configs"
    generated_config_dir.mkdir()

    rows = []
    for algorithm in ALGORITHMS:
        algorithm_config_path = generated_config_dir / f"{algorithm}.yaml"
        algorithm_config = _run_config(base_configs[algorithm], algorithm, run_dir)
        algorithm_config_path.write_text(
            yaml.safe_dump(algorithm_config, sort_keys=False),
            encoding="utf-8",
        )
        rows.append(_run_algorithm(algorithm, algorithm_config_path))

    write_jsonl(run_dir / "algorithm_summaries.jsonl", rows)
    summary = {
        "benchmark": "routing",
        "algorithms": list(ALGORITHMS),
        "configs": {key: str(path) for key, path in config_paths.items()},
        "rows": rows,
        "comparison": _comparison(rows),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")
    print(
        "summary: "
        + " ".join(
            f"{row['algorithm']}_avg_cost={row['average_cost_mean']:.6g}"
            for row in rows
        )
    )
    return 0


def _validate_base_configs(configs: dict[str, dict[str, Any]]) -> None:
    first_env = configs[ALGORITHMS[0]]["env"]
    first_bvi = configs[ALGORITHMS[0]].get("bvi", {})
    first_evaluation = configs[ALGORITHMS[0]].get("evaluation", {})
    first_policy_grid = configs[ALGORITHMS[0]].get("policy_grid", {})
    for expected_algorithm, config in configs.items():
        if config["env"]["name"] != "routing":
            raise ValueError(
                "routing comparison runner requires all env.name values to be 'routing'"
            )
        algorithm = config["algorithm"]["name"]
        if algorithm != expected_algorithm:
            raise ValueError(
                f"{expected_algorithm} config has algorithm.name={algorithm!r}"
            )
        if config["env"] != first_env:
            raise ValueError("routing comparison runner requires identical env configs")
        if config.get("bvi", {}) != first_bvi:
            raise ValueError("routing comparison runner requires identical bvi configs")
        if config.get("evaluation", {}) != first_evaluation:
            raise ValueError(
                "routing comparison runner requires identical evaluation configs"
            )
        if config.get("policy_grid", {}) != first_policy_grid:
            raise ValueError(
                "routing comparison runner requires identical policy_grid configs"
            )


def _comparison_config(
    config: dict[str, Any],
    algorithm: str,
    comparison_config: dict[str, Any],
) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    shared = comparison_config.get("shared", {})
    for section in ("env", "bvi", "evaluation", "policy_grid"):
        if section in shared:
            copied[section] = json.loads(json.dumps(shared[section]))
    copied.setdefault("algorithm", {})["name"] = algorithm
    return copied


def _run_config(
    config: dict[str, Any],
    algorithm: str,
    run_dir: Path,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    copied.setdefault("experiment", {})["name"] = f"routing_comparison_{algorithm}"
    copied["experiment"]["output_dir"] = str(run_dir / "children")
    return copied


def _run_algorithm(algorithm: str, config_path: Path) -> dict[str, Any]:
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
            f"{algorithm} failed with code {completed.returncode}:\n{completed.stderr}"
        )
    child_run_dir = _run_dir_from_stdout(completed.stdout)
    summary = json.loads((child_run_dir / "summary.json").read_text(encoding="utf-8"))
    return _row(algorithm, config_path, child_run_dir, summary, runtime)


def _row(
    algorithm: str,
    config_path: Path,
    child_run_dir: Path,
    summary: dict[str, Any],
    runtime: float,
) -> dict[str, Any]:
    evaluation = summary.get("evaluation", {})
    always_attack_evaluation = summary.get("always_attack_evaluation", {})
    minimax_evaluation = summary.get("minimax_evaluation", {})
    bvi_attacker_evaluation = summary.get("bvi_attacker_evaluation", {})
    policy_inspection = summary.get("policy_inspection", {})
    policy_comparison = summary.get("policy_comparison", {})
    q_diagnostic = summary.get("q_diagnostic", {})
    return {
        "algorithm": algorithm,
        "config": str(config_path),
        "run_dir": str(child_run_dir),
        "runtime_seconds": runtime,
        "average_cost_mean": evaluation.get("average_cost_mean"),
        "discounted_cost_mean": evaluation.get("discounted_cost_mean"),
        "always_attack_average_cost_mean": always_attack_evaluation.get(
            "average_cost_mean"
        ),
        "always_attack_discounted_cost_mean": always_attack_evaluation.get(
            "discounted_cost_mean"
        ),
        "minimax_average_cost_mean": minimax_evaluation.get("average_cost_mean"),
        "minimax_discounted_cost_mean": minimax_evaluation.get(
            "discounted_cost_mean"
        ),
        "bvi_attacker_average_cost_mean": bvi_attacker_evaluation.get(
            "average_cost_mean"
        ),
        "bvi_attacker_discounted_cost_mean": bvi_attacker_evaluation.get(
            "discounted_cost_mean"
        ),
        "defend_probability_mean": policy_inspection.get("defend_probability_mean"),
        "defend_states": policy_inspection.get(
            "num_states_p_defend_at_least_threshold"
        ),
        "policy_gap_mean": policy_comparison.get("p_defend_abs_gap_mean"),
        "q_reference_gap_mean": q_diagnostic.get("q_reference_abs_gap_mean"),
        "value_at_initial_state": summary.get("value_at_initial_state"),
        "training_steps": summary.get("total_steps"),
        "bvi_residual": summary.get("residual")
        or summary.get("bvi_reference", {}).get("residual"),
    }


def _comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: float(row["average_cost_mean"]))
    best_cost = float(ranked[0]["average_cost_mean"])
    best_algorithms = [
        row["algorithm"]
        for row in ranked
        if abs(float(row["average_cost_mean"]) - best_cost) <= 1e-12
    ]
    always_attack_ranked = sorted(
        rows, key=lambda row: float(row["always_attack_average_cost_mean"])
    )
    always_attack_best_cost = float(
        always_attack_ranked[0]["always_attack_average_cost_mean"]
    )
    always_attack_best_algorithms = [
        row["algorithm"]
        for row in always_attack_ranked
        if abs(
            float(row["always_attack_average_cost_mean"])
            - always_attack_best_cost
        )
        <= 1e-12
    ]
    minimax_ranked = sorted(
        rows, key=lambda row: float(row["minimax_average_cost_mean"])
    )
    minimax_best_cost = float(minimax_ranked[0]["minimax_average_cost_mean"])
    minimax_best_algorithms = [
        row["algorithm"]
        for row in minimax_ranked
        if abs(float(row["minimax_average_cost_mean"]) - minimax_best_cost) <= 1e-12
    ]
    bvi_attacker_ranked = sorted(
        rows, key=lambda row: float(row["bvi_attacker_average_cost_mean"])
    )
    bvi_attacker_best_cost = float(
        bvi_attacker_ranked[0]["bvi_attacker_average_cost_mean"]
    )
    bvi_attacker_best_algorithms = [
        row["algorithm"]
        for row in bvi_attacker_ranked
        if abs(
            float(row["bvi_attacker_average_cost_mean"]) - bvi_attacker_best_cost
        )
        <= 1e-12
    ]
    return {
        "random_attacker_ranked_by_average_cost": [
            row["algorithm"] for row in ranked
        ],
        "random_attacker_best_average_cost_algorithms": best_algorithms,
        "random_attacker_average_cost_gap_from_best": {
            row["algorithm"]: float(row["average_cost_mean"]) - best_cost
            for row in rows
        },
        "always_attack_ranked_by_average_cost": [
            row["algorithm"] for row in always_attack_ranked
        ],
        "always_attack_best_average_cost_algorithms": always_attack_best_algorithms,
        "always_attack_average_cost_gap_from_best": {
            row["algorithm"]: (
                float(row["always_attack_average_cost_mean"])
                - always_attack_best_cost
            )
            for row in rows
        },
        "minimax_ranked_by_average_cost": [
            row["algorithm"] for row in minimax_ranked
        ],
        "minimax_best_average_cost_algorithms": minimax_best_algorithms,
        "minimax_average_cost_gap_from_best": {
            row["algorithm"]: float(row["minimax_average_cost_mean"])
            - minimax_best_cost
            for row in rows
        },
        "bvi_attacker_ranked_by_average_cost": [
            row["algorithm"] for row in bvi_attacker_ranked
        ],
        "bvi_attacker_best_average_cost_algorithms": bvi_attacker_best_algorithms,
        "bvi_attacker_average_cost_gap_from_best": {
            row["algorithm"]: (
                float(row["bvi_attacker_average_cost_mean"])
                - bvi_attacker_best_cost
            )
            for row in rows
        },
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
