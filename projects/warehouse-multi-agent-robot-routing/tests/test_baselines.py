from warehouse_marl.baselines import greedy_joint_policy, reservation_policy
from warehouse_marl.environment import WarehouseRoutingEnv
from warehouse_marl.evaluate import evaluate_policy


def test_baseline_actions_are_valid():
    env = WarehouseRoutingEnv(seed=4)
    env.reset()
    for policy in (greedy_joint_policy, reservation_policy):
        action = policy(env)
        assert env.action_space.contains(action)


def test_reservation_policy_smoke_evaluation():
    metrics = evaluate_policy(reservation_policy, episodes=3, seed=10, n_robots=3, max_steps=50)
    assert 0.0 <= metrics.completion_rate <= 1.0
    assert metrics.mean_completed >= 0.0
    assert metrics.mean_collisions >= 0.0
    assert metrics.mean_steps > 0.0
