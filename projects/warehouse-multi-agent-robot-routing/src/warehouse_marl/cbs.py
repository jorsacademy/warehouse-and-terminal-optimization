from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
from typing import Iterable

import numpy as np

Position = tuple[int, int]
_MOVES: tuple[Position, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_MOVE_TO_ACTION = {move: i for i, move in enumerate(_MOVES)}


@dataclass(frozen=True)
class VertexConstraint:
    agent: int
    time: int
    cell: Position


@dataclass(frozen=True)
class EdgeConstraint:
    agent: int
    time: int
    source: Position
    target: Position


Constraint = VertexConstraint | EdgeConstraint


@dataclass(frozen=True)
class Conflict:
    agent1: int
    agent2: int
    time: int
    kind: str
    cell: Position | None = None
    edge1: tuple[Position, Position] | None = None
    edge2: tuple[Position, Position] | None = None


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(cell: Position, grid_size: int) -> Iterable[Position]:
    r, c = cell
    for dr, dc in _MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            yield (nr, nc)


def _agent_constraints(constraints: tuple[Constraint, ...], agent: int):
    vertex: set[tuple[int, Position]] = set()
    edge: set[tuple[int, Position, Position]] = set()
    max_time = 0
    for constraint in constraints:
        if constraint.agent != agent:
            continue
        max_time = max(max_time, constraint.time)
        if isinstance(constraint, VertexConstraint):
            vertex.add((constraint.time, constraint.cell))
        else:
            edge.add((constraint.time, constraint.source, constraint.target))
    return vertex, edge, max_time


def _reconstruct(
    parent: dict[tuple[Position, int], tuple[Position, int] | None],
    state: tuple[Position, int],
) -> list[Position]:
    path: list[Position] = []
    current: tuple[Position, int] | None = state
    while current is not None:
        path.append(current[0])
        current = parent[current]
    path.reverse()
    return path


def _low_level_path(
    start: Position,
    goal: Position,
    grid_size: int,
    constraints: tuple[Constraint, ...],
    agent: int,
    fixed: bool = False,
) -> list[Position] | None:
    vertex, edge, max_constraint_time = _agent_constraints(constraints, agent)
    if (0, start) in vertex:
        return None
    if fixed:
        if start != goal:
            return None
        if any(cell == start for _, cell in vertex):
            return None
        return [start]

    base_distance = _manhattan(start, goal)
    max_time = max(
        max_constraint_time + grid_size * grid_size + 2,
        base_distance + 4 * grid_size + max_constraint_time + 2,
    )
    serial = count()
    frontier: list[tuple[int, int, int, Position, int]] = []
    heapq.heappush(frontier, (base_distance, 0, next(serial), start, 0))
    parent: dict[tuple[Position, int], tuple[Position, int] | None] = {
        (start, 0): None
    }
    best_g: dict[tuple[Position, int], int] = {(start, 0): 0}

    while frontier:
        _, g, _, cell, time = heapq.heappop(frontier)
        state = (cell, time)
        if g != best_g.get(state):
            continue

        if cell == goal:
            if all(
                not (t >= time and constrained_cell == goal)
                for t, constrained_cell in vertex
            ):
                return _reconstruct(parent, state)

        if time >= max_time:
            continue

        next_time = time + 1
        for nxt in _neighbors(cell, grid_size):
            if (next_time, nxt) in vertex:
                continue
            if (next_time, cell, nxt) in edge:
                continue
            next_state = (nxt, next_time)
            next_g = g + 1
            if next_g >= best_g.get(next_state, 10**9):
                continue
            best_g[next_state] = next_g
            parent[next_state] = state
            heapq.heappush(
                frontier,
                (
                    next_g + _manhattan(nxt, goal),
                    next_g,
                    next(serial),
                    nxt,
                    next_time,
                ),
            )
    return None


def _position_at(path: list[Position], time: int) -> Position:
    return path[min(time, len(path) - 1)]


def _first_conflict(paths: list[list[Position]]) -> Conflict | None:
    horizon = max(len(path) for path in paths)
    for time in range(horizon):
        positions = [_position_at(path, time) for path in paths]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if positions[i] == positions[j]:
                    return Conflict(i, j, time, "vertex", cell=positions[i])
                if time > 0:
                    prev_i = _position_at(paths[i], time - 1)
                    prev_j = _position_at(paths[j], time - 1)
                    if prev_i == positions[j] and prev_j == positions[i]:
                        return Conflict(
                            i,
                            j,
                            time,
                            "edge",
                            edge1=(prev_i, positions[i]),
                            edge2=(prev_j, positions[j]),
                        )
    return None


def solve_cbs(
    starts: list[Position],
    goals: list[Position],
    grid_size: int,
    fixed_agents: set[int] | None = None,
    max_expansions: int = 10_000,
) -> list[list[Position]] | None:
    """Compute collision-free shortest paths with Conflict-Based Search.

    The high level branches on the first vertex or edge-swap conflict. The
    low level uses time-expanded A* and agents wait at their goal after arrival,
    matching standard MAPF semantics.
    """
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have equal length")
    if not starts:
        return []
    if len(set(starts)) != len(starts):
        raise ValueError("starts must be unique")
    fixed_agents = fixed_agents or set()

    root_constraints: tuple[Constraint, ...] = ()
    root_paths: list[list[Position]] = []
    for agent, (start, goal) in enumerate(zip(starts, goals)):
        path = _low_level_path(
            start,
            goal,
            grid_size,
            root_constraints,
            agent,
            fixed=agent in fixed_agents,
        )
        if path is None:
            return None
        root_paths.append(path)

    serial = count()
    queue: list[
        tuple[int, int, int, tuple[Constraint, ...], list[list[Position]]]
    ] = []

    def priority(paths: list[list[Position]]) -> tuple[int, int]:
        return sum(len(path) - 1 for path in paths), max(len(path) for path in paths)

    root_cost, root_makespan = priority(root_paths)
    heapq.heappush(
        queue,
        (root_cost, root_makespan, next(serial), root_constraints, root_paths),
    )

    expansions = 0
    while queue and expansions < max_expansions:
        _, _, _, constraints, paths = heapq.heappop(queue)
        expansions += 1
        conflict = _first_conflict(paths)
        if conflict is None:
            return paths

        branches: list[tuple[int, Constraint]]
        if conflict.kind == "vertex":
            assert conflict.cell is not None
            branches = [
                (
                    conflict.agent1,
                    VertexConstraint(
                        conflict.agent1,
                        conflict.time,
                        conflict.cell,
                    ),
                ),
                (
                    conflict.agent2,
                    VertexConstraint(
                        conflict.agent2,
                        conflict.time,
                        conflict.cell,
                    ),
                ),
            ]
        else:
            assert conflict.edge1 is not None and conflict.edge2 is not None
            branches = [
                (
                    conflict.agent1,
                    EdgeConstraint(
                        conflict.agent1,
                        conflict.time,
                        conflict.edge1[0],
                        conflict.edge1[1],
                    ),
                ),
                (
                    conflict.agent2,
                    EdgeConstraint(
                        conflict.agent2,
                        conflict.time,
                        conflict.edge2[0],
                        conflict.edge2[1],
                    ),
                ),
            ]

        for agent, new_constraint in branches:
            child_constraints = constraints + (new_constraint,)
            child_paths = [list(path) for path in paths]
            replanned = _low_level_path(
                starts[agent],
                goals[agent],
                grid_size,
                child_constraints,
                agent,
                fixed=agent in fixed_agents,
            )
            if replanned is None:
                continue
            child_paths[agent] = replanned
            cost, makespan = priority(child_paths)
            heapq.heappush(
                queue,
                (
                    cost,
                    makespan,
                    next(serial),
                    child_constraints,
                    child_paths,
                ),
            )
    return None


def _action_for_step(source: Position, target: Position) -> int:
    delta = (target[0] - source[0], target[1] - source[1])
    try:
        return _MOVE_TO_ACTION[delta]
    except KeyError as exc:
        raise RuntimeError(
            f"CBS produced a non-adjacent step: {source} -> {target}"
        ) from exc


def cbs_joint_policy(env) -> np.ndarray:
    """Receding-horizon CBS baseline for current pickup/drop-off targets.

    At each decision epoch, CBS computes collision-free shortest paths from
    current positions to current task targets and executes the first joint move.
    Completed robots are fixed in place and treated as persistent MAPF agents.
    """
    starts = [tuple(position) for position in env.positions]
    goals: list[Position] = []
    fixed_agents: set[int] = set()
    for i in range(env.n_robots):
        if int(env.phase[i]) == 2:
            goals.append(starts[i])
            fixed_agents.add(i)
        else:
            goals.append(tuple(env._target(i)))

    paths = solve_cbs(
        starts,
        goals,
        env.grid_size,
        fixed_agents=fixed_agents,
    )
    if paths is None:
        return np.zeros(env.n_robots, dtype=np.int64)

    actions = []
    for i, path in enumerate(paths):
        if len(path) < 2 or i in fixed_agents:
            actions.append(0)
        else:
            actions.append(_action_for_step(path[0], path[1]))
    return np.asarray(actions, dtype=np.int64)
