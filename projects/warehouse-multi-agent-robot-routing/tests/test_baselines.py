import numpy as np\n\nfrom warehouse_marl.baselines import cbs_joint_policy, greedy_joint_policy, reservation_policy\nfrom warehouse_marl.cbs import solve_cbs
from warehouse_marl.environment import WarehouseRoutingEnv
from warehouse_marl.evaluate import evaluate_policy


def test_baseline_actions_are_valid():
    env = WarehouseRoutingEnv(seed=4)
    env.reset()
    for policy in (greedy_joint_policy, reservation_policy, cbs_joint_policy):
        action = policy(env)
        assert env.action_space.contains(action)


def test_reservation_policy_smoke_evaluation():
    metrics = evaluate_policy(reservation_policy, episodes=3, seed=10, n_robots=3, max_steps=50)
    assert 0.0 <= metrics.completion_rate <= 1.0
    assert metrics.mean_completed >= 0.0
    assert metrics.mean_collisions >= 0.0
    assert metrics.mean_steps > 0.0



def test_cbs_resolves_vertex_conflict():
    paths = solve_cbs(
        starts=[(1, 0), (1, 2)],
        goals=[(1, 2), (1, 0)],
        grid_size=4,
    )
    assert paths is not None
    horizon = max(len(path) for path in paths)
    for t in range(horizon):
        positions = [path[min(t, len(path) - 1)] for path in paths]
        assert len(set(positions)) == len(positions)
        if t > 0:
            prev = [path[min(t - 1, len(path) - 1)] for path in paths]
            assert not (
                prev[0] == positions[1]
                and prev[1] == positions[0]
            )


def test_cbs_joint_policy_avoids_immediate_collision():
    env = WarehouseRoutingEnv(grid_size=4, n_robots=2, max_steps=20)
    env.reset(seed=3)
    env.positions = [(1, 0), (1, 2)]
    env.tasks = [
        type(env.tasks[0])((1, 2), (3, 3)),
        type(env.tasks[1])((1, 0), (0, 0)),
    ]
    env.phase = np.array([0, 0], dtype=np.int64)

    action = cbs_joint_policy(env)
    assert env.action_space.contains(action)
    _, _, _, _, info = env.step(action)
    assert info["collisions"] == 0
