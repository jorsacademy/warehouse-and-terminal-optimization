from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp


@dataclass(frozen=True)
class Block:
    equipment: str
    capacity_teu: float
    moves_per_hour: int
    has_power: bool
    has_safety: bool
    weight_limit_t: float
    gate_distance_m: float


class ContainerTerminalOptimizer:
    """MILP model for container yard block assignment and handling scheduling.

    The model jointly decides:
      * which yard block receives each container;
      * in which discrete time period the container is handled;
      * tardiness beyond the latest pickup/loading time.

    Yard capacity is modeled as a static planning-horizon capacity approximation.
    For a full terminal digital-twin model, replace it with time-indexed occupancy.
    """

    def __init__(
        self,
        n_containers: int = 150,
        planning_horizon: int = 72,
        period_hours: int = 2,
        seed: int = 42,
    ) -> None:
        if planning_horizon <= 0 or period_hours <= 0:
            raise ValueError("planning_horizon and period_hours must be positive.")
        if planning_horizon % period_hours != 0:
            raise ValueError("planning_horizon must be divisible by period_hours.")

        self.n_containers = n_containers
        self.planning_horizon = planning_horizon
        self.period_hours = period_hours
        self.time_periods = list(range(0, planning_horizon, period_hours))
        self.rng = np.random.default_rng(seed)

        self.blocks: Dict[int, Block] = {
            1: Block("RMG", 800, 25, True,  False, 50, 100),
            2: Block("RMG", 800, 25, True,  False, 50, 150),
            3: Block("RMG", 800, 25, False, False, 50, 200),
            4: Block("RTG", 600, 20, True,  False, 40, 250),
            5: Block("RTG", 600, 20, True,  False, 40, 300),
            6: Block("RTG", 600, 20, False, False, 40, 350),
            7: Block("RS",  300, 15, False, True,  35, 450),
            # Special-purpose block: supports containers that are both reefer
            # and hazardous, preventing silent loss of infeasible containers.
            8: Block("RS",  300, 15, True,  True,  45, 500),
        }

        self.storage_per_day = {"RMG": 12.50, "RTG": 8.75, "RS": 15.25}
        self.handling_per_move = {"RMG": 45.00, "RTG": 35.00, "RS": 55.00}
        self.delay_penalty_per_hour = 125.00
        self.repositioning_cost_per_100m = 15.00

        self.containers = self._generate_data()
        self._validate_instance()

    def _generate_data(self) -> pd.DataFrame:
        rows: List[dict] = []

        for i in range(self.n_containers):
            container_type = self.rng.choice(
                ["20ft", "40ft", "45ft"], p=[0.40, 0.50, 0.10]
            )

            if container_type == "20ft":
                weight = self.rng.normal(15, 5)
            elif container_type == "40ft":
                weight = self.rng.normal(25, 8)
            else:
                weight = self.rng.normal(28, 7)

            weight = float(np.clip(weight, 5, 45))
            is_reefer = bool(self.rng.random() < 0.12)
            is_hazardous = bool(self.rng.random() < 0.08)
            is_import = bool(self.rng.random() < 0.60)
            arrival = float(self.rng.uniform(0, 24))

            if is_import:
                earliest = arrival + float(self.rng.uniform(8, 24))
                latest = earliest + float(self.rng.uniform(12, 48))
            else:
                earliest = arrival + float(self.rng.uniform(4, 16))
                latest = earliest + float(self.rng.uniform(8, 24))

            # Keep the service window inside the modeled horizon.
            earliest = min(earliest, self.planning_horizon - self.period_hours)
            latest = min(
                max(latest, earliest + self.period_hours),
                self.planning_horizon,
            )

            expected_dwell = max(
                self.period_hours,
                (latest - arrival) * float(self.rng.uniform(0.60, 0.90)),
            )

            rows.append(
                {
                    "container_id": f"CONT_{i + 1:03d}",
                    "type": container_type,
                    "weight": weight,
                    "is_reefer": is_reefer,
                    "is_hazardous": is_hazardous,
                    "is_import": is_import,
                    "arrival_time": arrival,
                    "earliest_pickup": earliest,
                    "latest_pickup": latest,
                    "expected_dwell_time": expected_dwell,
                }
            )

        df = pd.DataFrame(rows)
        df["teu_equivalent"] = df["type"].map(
            {"20ft": 1.0, "40ft": 2.0, "45ft": 2.25}
        )
        df["priority"] = pd.cut(
            df["latest_pickup"],
            bins=[-np.inf, 24, 48, np.inf],
            labels=["HIGH", "MEDIUM", "LOW"],
            right=False,
        ).astype(str)
        return df

    def _compatible(self, i: int, j: int) -> bool:
        c = self.containers.iloc[i]
        b = self.blocks[j]
        if c["is_reefer"] and not b.has_power:
            return False
        if c["is_hazardous"] and not b.has_safety:
            return False
        if c["weight"] > b.weight_limit_t:
            return False
        return True

    def _feasible_periods(self, i: int) -> List[int]:
        c = self.containers.iloc[i]
        lower_bound = max(float(c["arrival_time"]), float(c["earliest_pickup"]))
        return [t for t in self.time_periods if t >= lower_bound]

    def _validate_instance(self) -> None:
        no_block = [
            self.containers.iloc[i]["container_id"]
            for i in range(self.n_containers)
            if not any(self._compatible(i, j) for j in self.blocks)
        ]
        if no_block:
            raise ValueError(f"No compatible yard block for: {no_block}")

        no_period = [
            self.containers.iloc[i]["container_id"]
            for i in range(self.n_containers)
            if not self._feasible_periods(i)
        ]
        if no_period:
            raise ValueError(f"No feasible handling period for: {no_period}")

    def build_model(self) -> None:
        model = pulp.LpProblem("Container_Terminal_Yard_Optimization", pulp.LpMinimize)

        x: Dict[Tuple[int, int], pulp.LpVariable] = {}
        y: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
        delay: Dict[int, pulp.LpVariable] = {}

        for i in range(self.n_containers):
            for j in self.blocks:
                if self._compatible(i, j):
                    x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
                    for t in self._feasible_periods(i):
                        y[i, j, t] = pulp.LpVariable(
                            f"y_{i}_{j}_{t}", cat="Binary"
                        )
            delay[i] = pulp.LpVariable(f"delay_{i}", lowBound=0)

        objective_terms = []
        for i in range(self.n_containers):
            c = self.containers.iloc[i]
            dwell_days = float(c["expected_dwell_time"]) / 24.0

            for j, b in self.blocks.items():
                if (i, j) not in x:
                    continue

                objective_terms.append(
                    self.storage_per_day[b.equipment] * dwell_days * x[i, j]
                )
                objective_terms.append(
                    self.handling_per_move[b.equipment] * x[i, j]
                )
                objective_terms.append(
                    self.repositioning_cost_per_100m
                    * (b.gate_distance_m / 100.0)
                    * x[i, j]
                )

            objective_terms.append(self.delay_penalty_per_hour * delay[i])

        model += pulp.lpSum(objective_terms)

        # Every container must be assigned exactly once.
        for i in range(self.n_containers):
            model += (
                pulp.lpSum(x[i, j] for j in self.blocks if (i, j) in x) == 1,
                f"assignment_{i}",
            )

        # Static yard capacity approximation in TEU.
        for j, b in self.blocks.items():
            model += (
                pulp.lpSum(
                    float(self.containers.iloc[i]["teu_equivalent"]) * x[i, j]
                    for i in range(self.n_containers)
                    if (i, j) in x
                )
                <= b.capacity_teu,
                f"yard_capacity_{j}",
            )

        # Exactly one handling slot for the selected block.
        for i in range(self.n_containers):
            for j in self.blocks:
                if (i, j) not in x:
                    continue
                model += (
                    pulp.lpSum(
                        y[i, j, t]
                        for t in self._feasible_periods(i)
                        if (i, j, t) in y
                    )
                    == x[i, j],
                    f"handling_link_{i}_{j}",
                )

        # Equipment throughput in each time bucket.
        for j, b in self.blocks.items():
            max_moves = b.moves_per_hour * self.period_hours
            for t in self.time_periods:
                model += (
                    pulp.lpSum(
                        y[i, j, t]
                        for i in range(self.n_containers)
                        if (i, j, t) in y
                    )
                    <= max_moves,
                    f"equipment_capacity_{j}_{t}",
                )

        # Tardiness is tied directly to the chosen handling period.
        for i in range(self.n_containers):
            service_time = pulp.lpSum(
                t * var
                for (ii, _j, t), var in y.items()
                if ii == i
            )
            latest = float(self.containers.iloc[i]["latest_pickup"])
            model += delay[i] >= service_time - latest, f"delay_def_{i}"

        self.model = model
        self.x = x
        self.y = y
        self.delay = delay

    def solve(self, time_limit_seconds: int = 120, relative_gap: float = 0.01) -> bool:
        if not hasattr(self, "model"):
            self.build_model()

        solver = pulp.PULP_CBC_CMD(
            msg=True,
            timeLimit=time_limit_seconds,
            gapRel=relative_gap,
        )
        self.model.solve(solver)
        self.status = pulp.LpStatus[self.model.status]

        if self.status not in {"Optimal"}:
            print(f"Solver status: {self.status}")
            return False

        self._extract_solution()
        return True

    def _extract_solution(self) -> None:
        rows = []

        for i in range(self.n_containers):
            selected_block = next(
                j
                for j in self.blocks
                if (i, j) in self.x and pulp.value(self.x[i, j]) > 0.5
            )
            selected_time = next(
                t
                for t in self._feasible_periods(i)
                if (i, selected_block, t) in self.y
                and pulp.value(self.y[i, selected_block, t]) > 0.5
            )

            c = self.containers.iloc[i]
            rows.append(
                {
                    "container_id": c["container_id"],
                    "container_idx": i,
                    "assigned_block": selected_block,
                    "block_type": self.blocks[selected_block].equipment,
                    "service_time": selected_time,
                    "delay": float(pulp.value(self.delay[i]) or 0.0),
                    "teu_equivalent": float(c["teu_equivalent"]),
                    "is_import": bool(c["is_import"]),
                    "is_reefer": bool(c["is_reefer"]),
                    "is_hazardous": bool(c["is_hazardous"]),
                    "weight": float(c["weight"]),
                    "priority": c["priority"],
                }
            )

        self.solution_df = pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        if not hasattr(self, "solution_df"):
            raise RuntimeError("Solve the model before requesting a summary.")

        rows = []
        for j, b in self.blocks.items():
            subset = self.solution_df[self.solution_df["assigned_block"] == j]
            used_teu = subset["teu_equivalent"].sum()
            rows.append(
                {
                    "block": j,
                    "equipment": b.equipment,
                    "containers": len(subset),
                    "used_teu": used_teu,
                    "capacity_teu": b.capacity_teu,
                    "utilization_pct": 100.0 * used_teu / b.capacity_teu,
                    "avg_delay_h": subset["delay"].mean() if len(subset) else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def plot_summary(self) -> None:
        summary = self.summary()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(summary["block"], summary["utilization_pct"])
        ax.set_xlabel("Yard block")
        ax.set_ylabel("TEU utilization (%)")
        ax.set_title("Container terminal yard utilization")
        ax.set_xticks(summary["block"])
        fig.tight_layout()
        plt.show()


def main() -> None:
    optimizer = ContainerTerminalOptimizer()
    optimizer.build_model()

    if not optimizer.solve():
        raise SystemExit("No optimal solution returned by CBC.")

    print(f"Status: {optimizer.status}")
    print(f"Objective: ${pulp.value(optimizer.model.objective):,.2f}")
    print(f"Assigned containers: {len(optimizer.solution_df)}")
    print(f"Total delay: {optimizer.solution_df['delay'].sum():.2f} h")
    print()
    print(optimizer.summary().to_string(index=False))

    optimizer.plot_summary()


if __name__ == "__main__":
    main()
