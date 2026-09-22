from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict

import numpy as np

from .baselines import greedy_joint_policy, reservation_policy
from .environment import WarehouseRoutingEnv


@dataclass
class Metrics:
    mean_reward: float
    mean_completed: float
    completion_rate: float
    mean_collisions: float
    mean_steps: float
    mean_throughput: float


def evaluate_policy(
    policy: Callable[[WarehouseRoutingEnv], np.ndarray],
    episodes: int = 50,
    seed: int = 7,
    **env_kwargs,
) -> Metrics:
    rewards, completed, collisions, steps, throughput = [], [], [], [], []
    successes = 0

    for ep in range(episodes):
        env = WarehouseRoutingEnv(seed=seed + ep, **env_kwargs)
        _, info = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(env)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        completed.append(info["completed_deliveries"])
        collisions.append(info["collisions"])
        steps.append(info["steps"])
        throughput.append(info["throughput"])
        successes += int(info["completed_deliveries"] == env.n_robots)

    return Metrics(
        mean_reward=float(np.mean(rewards)),
        mean_completed=float(np.mean(completed)),
        completion_rate=successes / episodes,
        mean_collisions=float(np.mean(collisions)),
        mean_steps=float(np.mean(steps)),
        mean_throughput=float(np.mean(throughput)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--robots", type=int, default=3)
    args = parser.parse_args()

    policies: Dict[str, Callable] = {
        "greedy": greedy_joint_policy,
        "reservation": reservation_policy,
    }
    for name, policy in policies.items():
        metrics = evaluate_policy(policy, episodes=args.episodes, n_robots=args.robots)
        print(name, asdict(metrics))


if __name__ == "__main__":
    main()
