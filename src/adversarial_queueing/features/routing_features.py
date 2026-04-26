"""Feature maps for the routing benchmark."""

from __future__ import annotations

import numpy as np


def routing_features(
    state: tuple[int, ...],
    attacker_action: int,
    defender_action: int,
    feature_set: str = "basic",
) -> np.ndarray:
    """Return ``phi(x, a, b)`` for routing AMQ prototypes."""

    x = np.array(state, dtype=float)
    a = float(attacker_action)
    b = float(defender_action)

    if feature_set == "basic":
        return np.concatenate(
            [
                np.array(
                    [
                        1.0,
                        float(x.sum()),
                        float(np.dot(x, x)),
                        float(x.max() - x.min()) if x.size else 0.0,
                        a,
                        b,
                        a * b,
                    ],
                    dtype=float,
                ),
                x,
            ]
        )

    raise ValueError(f"unknown routing feature_set: {feature_set}")


def routing_feature_dim(num_queues: int, feature_set: str = "basic") -> int:
    return int(
        routing_features(
            state=tuple(0 for _ in range(num_queues)),
            attacker_action=0,
            defender_action=0,
            feature_set=feature_set,
        ).shape[0]
    )
