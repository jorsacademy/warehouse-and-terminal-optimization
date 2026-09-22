from __future__ import annotations

from typing import Iterable

import numpy as np

from .environment import WarehouseRoutingEnv


def _greedy_action(position, target) -> int:
    r, c = position
    tr, tc = target
    if r < tr:
        return 2
    if r > tr:
        return 1
    if c < tc:
        return 4
    if c > tc:
        return 3
    return 0


def greedy_joint_policy(env: WarehouseRoutingEnv) -> np.ndarray:
    """Move every unfinished robot greedily toward its current target."""
    actions = []
    for i in range(env.n_robots):
        if env.phase[i] == 2:
            actions.append(0)
        else:
            actions.append(_greedy_action(env.positions[i], env._target(i)))
    return np.asarray(actions, dtype=np.int64)


def reservation_policy(env: WarehouseRoutingEnv) -> np.ndarray:
    """Greedy routing with one-step vertex reservation.

    Lower-index robots receive priority. This simple baseline approximates a
    decentralized warehouse traffic rule and usually reduces collisions versus
    fully independent greedy motion.
    """
    chosen = []
    reserved = set()
    for i in range(env.n_robots):
        if env.phase[i] == 2:
            chosen.append(0)
            reserved.add(env.positions[i])
            continue

        preferred = _greedy_action(env.positions[i], env._target(i))
        candidates: Iterable[int] = [preferred, 0, 1, 2, 3, 4]
        action = 0
        for candidate in candidates:
            proposal = env._move(env.positions[i], candidate)
            if proposal not in reserved:
                action = candidate
                reserved.add(proposal)
                break
        chosen.append(action)
    return np.asarray(chosen, dtype=np.int64)
