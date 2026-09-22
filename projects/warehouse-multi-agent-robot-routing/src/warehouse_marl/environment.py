from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ACTIONS = {
    0: (0, 0),   # wait
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}


@dataclass(frozen=True)
class Task:
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]


class WarehouseRoutingEnv(gym.Env):
    """Small cooperative warehouse-routing benchmark.

    Each robot receives a task and must visit pickup then drop-off. The joint
    action contains one discrete movement action per robot. Rewards penalize
    travel, congestion/collisions, and unfinished work while rewarding
    completed deliveries.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 8,
        n_robots: int = 3,
        max_steps: int = 80,
        collision_penalty: float = 5.0,
        completion_reward: float = 25.0,
        step_cost: float = 0.2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if n_robots < 2:
            raise ValueError("n_robots must be at least 2")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")

        self.grid_size = grid_size
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.collision_penalty = collision_penalty
        self.completion_reward = completion_reward
        self.step_cost = step_cost
        self._initial_seed = seed

        self.action_space = spaces.MultiDiscrete(np.full(n_robots, 5, dtype=np.int64))
        # robot_x, robot_y, target_x, target_y, phase for each robot + normalized time
        low = np.zeros(n_robots * 5 + 1, dtype=np.float32)
        high = np.ones(n_robots * 5 + 1, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.positions: List[Tuple[int, int]] = []
        self.tasks: List[Task] = []
        self.phase = np.zeros(n_robots, dtype=np.int64)  # 0 pickup, 1 dropoff, 2 complete
        self.steps = 0
        self.collisions = 0
        self.completed_deliveries = 0

    def _sample_unique_cells(self, count: int) -> List[Tuple[int, int]]:
        total = self.grid_size * self.grid_size
        if count > total:
            raise ValueError("Not enough grid cells for requested entities")
        idx = self.np_random.choice(total, size=count, replace=False)
        return [(int(i // self.grid_size), int(i % self.grid_size)) for i in idx]

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=self._initial_seed if seed is None else seed)
        cells = self._sample_unique_cells(self.n_robots * 3)
        self.positions = cells[: self.n_robots]
        pickups = cells[self.n_robots : 2 * self.n_robots]
        dropoffs = cells[2 * self.n_robots : 3 * self.n_robots]
        self.tasks = [Task(p, d) for p, d in zip(pickups, dropoffs)]
        self.phase[:] = 0
        self.steps = 0
        self.collisions = 0
        self.completed_deliveries = 0
        return self._observation(), self._info()

    def _target(self, i: int) -> Tuple[int, int]:
        if self.phase[i] == 0:
            return self.tasks[i].pickup
        return self.tasks[i].dropoff

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dr, dc = ACTIONS[int(action)]
        r = int(np.clip(pos[0] + dr, 0, self.grid_size - 1))
        c = int(np.clip(pos[1] + dc, 0, self.grid_size - 1))
        return (r, c)

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.n_robots,):
            raise ValueError(f"Expected action shape {(self.n_robots,)}, got {action.shape}")

        proposals = [self._move(p, int(a)) for p, a in zip(self.positions, action)]
        reward = -self.step_cost * float(np.count_nonzero(action))

        # Vertex collisions: robots proposing the same cell remain in place.
        counts: Dict[Tuple[int, int], int] = {}
        for p in proposals:
            counts[p] = counts.get(p, 0) + 1
        collided = {p for p, c in counts.items() if c > 1}

        # Edge swaps are also treated as collisions.
        swap_flags = set()
        for i in range(self.n_robots):
            for j in range(i + 1, self.n_robots):
                if proposals[i] == self.positions[j] and proposals[j] == self.positions[i]:
                    swap_flags.update((i, j))

        new_positions = list(self.positions)
        collision_events = 0
        for i, proposal in enumerate(proposals):
            if proposal in collided or i in swap_flags:
                collision_events += 1
                continue
            new_positions[i] = proposal

        if collision_events:
            self.collisions += collision_events
            reward -= self.collision_penalty * collision_events
        self.positions = new_positions

        for i in range(self.n_robots):
            if self.phase[i] == 0 and self.positions[i] == self.tasks[i].pickup:
                self.phase[i] = 1
                reward += 2.0
            if self.phase[i] == 1 and self.positions[i] == self.tasks[i].dropoff:
                self.phase[i] = 2
                self.completed_deliveries += 1
                reward += self.completion_reward

        self.steps += 1
        terminated = bool(np.all(self.phase == 2))
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            unfinished = int(np.count_nonzero(self.phase != 2))
            reward -= 2.0 * unfinished
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _observation(self) -> np.ndarray:
        denom = float(self.grid_size - 1)
        obs: List[float] = []
        for i in range(self.n_robots):
            r, c = self.positions[i]
            if self.phase[i] == 2:
                tr, tc = r, c
            else:
                tr, tc = self._target(i)
            obs.extend([r / denom, c / denom, tr / denom, tc / denom, self.phase[i] / 2.0])
        obs.append(min(self.steps / self.max_steps, 1.0))
        return np.asarray(obs, dtype=np.float32)

    def _info(self) -> dict:
        return {
            "completed_deliveries": self.completed_deliveries,
            "collisions": self.collisions,
            "steps": self.steps,
            "throughput": self.completed_deliveries / max(self.steps, 1),
        }

    def render(self):
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        for i, (r, c) in enumerate(self.positions):
            grid[r][c] = str(i)
        return "\n".join(" ".join(row) for row in grid)
