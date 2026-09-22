from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cooperative PPO/MAPPO-style policies with RLlib")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    from .rllib_env import WarehouseMultiAgentEnv

    env_name = "warehouse_multi_agent_v0"
    register_env(env_name, lambda config: WarehouseMultiAgentEnv(config))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    probe = WarehouseMultiAgentEnv({"n_robots": args.robots, "grid_size": args.grid_size})
    obs_space = probe.observation_spaces[probe.agents[0]]
    act_space = probe.action_spaces[probe.agents[0]]

    policies = {"shared_policy": (None, obs_space, act_space, {}) }
    config = (
        PPOConfig()
        .environment(env_name, env_config={"n_robots": args.robots, "grid_size": args.grid_size})
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *a, **k: "shared_policy",
        )
    )

    algo = config.build()
    try:
        for i in range(args.iterations):
            result = algo.train()
            reward = result.get("env_runners", {}).get("episode_return_mean")
            if reward is None:
                reward = result.get("episode_reward_mean")
            print(f"iteration={i + 1} mean_episode_reward={reward}")
        checkpoint = algo.save()
        print(f"checkpoint={checkpoint}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
