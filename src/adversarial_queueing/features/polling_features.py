"""Feature maps for the polling benchmark."""

from __future__ import annotations

import numpy as np


def polling_features(
    state: tuple[int, ...],
    attacker_action: int,
    defender_action: int,
    feature_set: str = "basic",
) -> np.ndarray:
    """Return ``phi(s, a, b)`` for polling AMQ smoke experiments."""

    values = tuple(int(value) for value in state)
    queues = np.array(values[:-1], dtype=float)
    position = int(values[-1])
    a = float(attacker_action)
    b = float(defender_action)
    if feature_set == "basic":
        return np.concatenate(
            [
                np.array(
                    [
                        1.0,
                        float(queues.sum()),
                        float(np.dot(queues, queues)),
                        float(queues.max() - queues.min()) if queues.size else 0.0,
                        float(position),
                        a,
                        b,
                        a * b,
                    ],
                    dtype=float,
                ),
                queues,
            ]
        )
    if feature_set == "action_interaction":
        base = _polling_state_basis(queues, position)
        joint_action = attacker_action * 2 + defender_action
        joint = _one_hot(joint_action, 4)
        return np.outer(joint, base).ravel()
    raise ValueError(f"unknown polling feature_set: {feature_set}")


def polling_feature_dim(num_queues: int, feature_set: str = "basic") -> int:
    state = tuple(0 for _ in range(num_queues)) + (0,)
    return int(
        polling_features(
            state=state,
            attacker_action=0,
            defender_action=0,
            feature_set=feature_set,
        ).shape[0]
    )


def _polling_state_basis(queues: np.ndarray, position: int) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [
                    1.0,
                    float(queues.sum()),
                    float(np.dot(queues, queues)),
                    float(queues.max() - queues.min()) if queues.size else 0.0,
                    float(position),
                ],
                dtype=float,
            ),
            queues,
        ]
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    if not 0 <= index < size:
        raise ValueError(f"index {index} outside one-hot size {size}")
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out
