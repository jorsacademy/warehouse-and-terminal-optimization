# Warehouse Multi-Agent Robot Routing

A reproducible industrial-engineering benchmark for cooperative routing of autonomous mobile robots (AMRs/AGVs) in a warehouse. The project focuses on a practical combination of **task execution, congestion avoidance, collision handling, throughput, and multi-agent reinforcement learning** rather than shortest-path routing alone.

## Industrial problem

Modern warehouses operate fleets of mobile robots that repeatedly travel to pickup and drop-off locations. A locally shortest route can be globally poor when many robots compete for the same aisle or intersection. The control problem is therefore sequential and coupled: every robot action changes the traffic conditions experienced by the rest of the fleet.

This repository models that problem as a cooperative multi-agent Markov decision process.

### State

For each robot the normalized global observation contains:

- robot row and column,
- current target row and column,
- task phase: pickup, delivery, or complete,
- normalized episode time.

The RLlib adapter gives each homogeneous robot the same global state, which supports centralized-training/decentralized-action experiments.

### Actions

Each robot chooses one discrete action per decision epoch:

0. wait
1. move up
2. move down
3. move left
4. move right

The joint environment executes all robot actions simultaneously.

### Reward

The cooperative reward combines:

- movement cost,
- pickup progress reward,
- delivery completion reward,
- vertex-collision penalty,
- edge-swap collision penalty,
- unfinished-task penalty at the time horizon.

This creates the core trade-off between aggressive shortest-path motion and globally coordinated congestion avoidance.

## Algorithms and baselines

### Greedy routing

Every robot independently moves along a Manhattan-shortest direction toward its current pickup/drop-off target. This is a useful lower-complexity benchmark but can generate congestion and collisions.

### Reservation heuristic

A simple one-step traffic reservation rule gives lower-index robots priority over proposed cells. It approximates warehouse right-of-way logic and demonstrates how coordination can improve over independent greedy routing without learning.

### MAPPO-style cooperative PPO

`WarehouseMultiAgentEnv` adapts the joint dynamics to RLlib's `MultiAgentEnv` interface. `train_mappo.py` trains a shared PPO policy across homogeneous robots. Because all agents receive a global observation and share a cooperative reward, the setup is suitable for MAPPO-style centralized-training experiments.

> Strict MAPPO implementations often use a centralized critic with decentralized actor observations. RLlib PPO with a shared global-observation policy is deliberately described here as **MAPPO-style** rather than claiming an exact reproduction of a specific paper implementation.

## KPIs

The evaluation layer reports:

- mean episode reward,
- mean completed deliveries,
- full-fleet completion rate,
- mean collisions,
- mean episode steps,
- throughput = completed deliveries / elapsed steps.

For industrial use, collision rate and throughput should be evaluated together. A controller that finishes tasks quickly but creates excessive conflict is not operationally acceptable.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── warehouse_marl/
│       ├── __init__.py
│       ├── environment.py
│       ├── baselines.py
│       ├── evaluate.py
│       ├── rllib_env.py
│       └── train_mappo.py
├── tests/
│   ├── test_environment.py
│   └── test_baselines.py
└── .github/workflows/ci.yml
```

## Installation

Core benchmark and tests:

```bash
pip install -e '.[test]'
```

RLlib training dependencies:

```bash
pip install -e '.[rl]'
```

## Run baseline experiments

```bash
python -m warehouse_marl.evaluate --episodes 100 --robots 3
```

This compares independent greedy routing with the reservation heuristic using identical stochastic warehouse instances.

## Train cooperative PPO / MAPPO-style policy

```bash
python -m warehouse_marl.train_mappo --iterations 20 --robots 3 --grid-size 8
```

The training script uses one shared policy for homogeneous robots and saves an RLlib checkpoint after training.

## Modeling assumptions

This benchmark intentionally remains compact enough for experimentation and CI. It currently assumes:

- a rectangular grid,
- one pickup/drop-off task per robot per episode,
- homogeneous robots,
- simultaneous movement decisions,
- no static shelf obstacles,
- shared cooperative reward.

These assumptions make the model transparent while preserving the central multi-agent coordination problem.

## Research extensions

High-value extensions for academic or industrial work include:

1. static shelf/aisle topology and A* path planning,
2. dynamic order arrivals and online task allocation,
3. battery state and charging-station scheduling,
4. heterogeneous robot capacities and speeds,
5. congestion-sensitive travel times,
6. prioritized experience or curriculum learning as fleet size grows,
7. centralized critic with local actor observations for a stricter MAPPO implementation,
8. comparison with CBS, prioritized planning, min-cost flow, or OR-Tools assignment baselines,
9. digital-twin integration using live WMS/WES events,
10. safety constraints and deadlock detection.

## Industrial-engineering interpretation

This project sits at the intersection of operations research, warehouse control, robotics, simulation, and reinforcement learning. The important question is not whether RL can move robots through a grid; it is whether a learned policy improves **system-level throughput and congestion performance** against transparent dispatching/routing baselines under uncertainty.
