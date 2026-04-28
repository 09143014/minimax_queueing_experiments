#!/usr/bin/env python3
"""Run polling BVI/AMQ/NNQ smoke configs and aggregate summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.utils.config import load_config
from adversarial_queueing.utils.output import create_run_dir, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to comparison YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_dir = create_run_dir(
        config["experiment"].get("output_dir", "results"),
        config["experiment"].get("name", "polling_comparison"),
    )
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")

    rows = []
    for run in config["runs"]:
        row = _run_one_method(str(run["method"]), Path(run["config"]))
        rows.append(row)

    write_jsonl(run_dir / "comparison.jsonl", rows)
    summary = {
        "benchmark": "polling",
        "num_methods": len(rows),
        "methods": [row["method"] for row in rows],
        "rows": rows,
        "best_average_cost_methods": _best_methods(rows),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")
    for row in rows:
        print(
            "method_summary: "
            f"method={row['method']} "
            f"average_cost_mean={row['average_cost_mean']:.6g} "
            f"runtime_seconds={row['runtime_seconds']:.3f}"
        )
    return 0


def _run_one_method(method: str, config_path: Path) -> dict:
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
            f"{method} run failed with code {completed.returncode}:\n{completed.stderr}"
        )
    child_run_dir = _run_dir_from_stdout(completed.stdout)
    summary = json.loads((child_run_dir / "summary.json").read_text(encoding="utf-8"))
    evaluation = summary["evaluation"]
    return {
        "method": method,
        "config": str(config_path),
        "run_dir": str(child_run_dir),
        "runtime_seconds": runtime,
        "average_cost_mean": evaluation["average_cost_mean"],
        "discounted_cost_mean": evaluation["discounted_cost_mean"],
        "tail_fraction_mean": evaluation["tail_fraction_mean"],
        "boundary_hit_fraction_mean": evaluation["boundary_hit_fraction_mean"],
    }


def _best_methods(rows: list[dict]) -> list[str]:
    best = min(float(row["average_cost_mean"]) for row in rows)
    return [
        str(row["method"])
        for row in rows
        if float(row["average_cost_mean"]) == best
    ]


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
