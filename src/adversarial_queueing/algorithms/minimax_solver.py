"""One-state zero-sum matrix-game solver."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_zero_sum_matrix_game(payoff: np.ndarray, player: str = "defender") -> dict:
    """Solve ``min_defender max_attacker payoff[a, b]``.

    Args:
        payoff: Defender costs / attacker rewards with shape
            ``[num_attacker_actions, num_defender_actions]``.
        player: Currently only ``"defender"`` is supported.
    """

    if player != "defender":
        raise ValueError("only player='defender' is supported")

    matrix = np.asarray(payoff, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("payoff must be a 2D array")

    num_attacker, num_defender = matrix.shape
    objective = np.zeros(num_defender + 1)
    objective[-1] = 1.0

    # Variables are defender probabilities sigma[0:B] and value c.
    a_ub = np.column_stack([matrix, -np.ones(num_attacker)])
    b_ub = np.zeros(num_attacker)
    a_eq = np.zeros((1, num_defender + 1))
    a_eq[0, :num_defender] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * num_defender + [(None, None)]

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"matrix-game LP failed: {result.message}")

    defender_strategy = np.asarray(result.x[:num_defender], dtype=float)
    defender_strategy = np.clip(defender_strategy, 0.0, 1.0)
    defender_strategy = defender_strategy / defender_strategy.sum()
    value = float(result.x[-1])

    attacker_payoffs = matrix @ defender_strategy
    best_attacker = np.isclose(attacker_payoffs, attacker_payoffs.max(), atol=1e-9)
    attacker_strategy = best_attacker.astype(float)
    attacker_strategy /= attacker_strategy.sum()

    return {
        "value": value,
        "attacker_strategy": attacker_strategy,
        "defender_strategy": defender_strategy,
    }

