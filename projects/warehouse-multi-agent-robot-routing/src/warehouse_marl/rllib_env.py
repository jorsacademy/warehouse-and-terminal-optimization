from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np

from .environment import WarehouseRoutingEnv

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # keeps the core package importable without RLlib
    class MultiAgentEnv:  # type: ignore
        pass


class WarehouseMultiAgentEnv(MultiAgentEnv):
    """RLlib multi-agent adapter around the warehouse joint dynamics.

    Each robot observes the global normalized state. This centralized-training
    friendly representation is suitable for cooperative PPO/MAPPO-style
    experiments while execution remains one discrete movement action per robot.
    """

    def __init__(self, config=None):
        config = config or {}
        self.core = WarehouseRoutingEnv(
            grid_size=int(config.get("grid_size", 8)),
            n_robots=int(config.get("n_robots", 3)),
            max_steps=int(config.get("max_steps", 80)),
            seed=config.get("seed"),
        )
        self.agents = [f"robot_{i}" for i in range(self.core.n_robots)]
        self.possible_agents = list(self.agents)
        self.observation_spaces = {a: self.core.observation_space for a in self.agents}
        self.action_spaces = {a: gym.spaces.Discrete(5) for a in self.agents}

    def reset(self, *, seed=None, options=None):
        obs, info = self.core.reset(seed=seed, options=options)
        observations = {a: obs.copy() for a in self.agents}
        infos = {a: dict(info) for a in self.agents}
        return observations, infos

    def step(self, action_dict: Dict[str, int]):
        joint = np.asarray([action_dict.get(a, 0) for a in self.agents], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.core.step(joint)
        observations = {a: obs.copy() for a in self.agents}
        # Cooperative shared reward.
        rewards = {a: reward for a in self.agents}
        terminateds = {a: terminated for a in self.agents}
        truncateds = {a: truncated for a in self.agents}
        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated
        infos = {a: dict(info) for a in self.agents}
        return observations, rewards, terminateds, truncateds, infos

    def observation_space_sample(self, agent_ids=None):
        ids = agent_ids or self.agents
        return {a: self.observation_spaces[a].sample() for a in ids}

    def action_space_sample(self, agent_ids=None):
        ids = agent_ids or self.agents
        return {a: self.action_spaces[a].sample() for a in ids}
