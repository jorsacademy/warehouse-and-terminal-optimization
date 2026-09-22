import numpy as np
from gymnasium.utils.env_checker import check_env

from warehouse_marl.environment import Task, WarehouseRoutingEnv


def test_gymnasium_contract():
    env = WarehouseRoutingEnv(seed=1)
    check_env(env, skip_render_check=True)


def test_reset_is_deterministic_with_seed():
    env = WarehouseRoutingEnv()
    obs1, _ = env.reset(seed=42)
    positions1 = list(env.positions)
    tasks1 = list(env.tasks)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)
    assert positions1 == env.positions
    assert tasks1 == env.tasks


def test_vertex_collision_keeps_robots_in_place_and_penalizes():
    env = WarehouseRoutingEnv(grid_size=5, n_robots=2, collision_penalty=7.0, seed=2)
    env.reset()
    env.positions = [(2, 1), (2, 3)]
    env.tasks = [Task((4, 4), (0, 0)), Task((4, 0), (0, 4))]
    before = list(env.positions)
    _, reward, _, _, info = env.step(np.array([4, 3]))
    assert env.positions == before
    assert info["collisions"] == 2
    assert reward <= -14.0


def test_delivery_completion_updates_kpis():
    env = WarehouseRoutingEnv(grid_size=5, n_robots=2, seed=3)
    env.reset()
    env.positions = [(1, 1), (4, 4)]
    env.tasks = [Task((1, 2), (1, 3)), Task((0, 0), (0, 1))]
    env.phase[:] = [0, 2]
    env.completed_deliveries = 1
    env.step(np.array([4, 0]))
    assert env.phase[0] == 1
    _, _, terminated, _, info = env.step(np.array([4, 0]))
    assert terminated
    assert info["completed_deliveries"] == 2
