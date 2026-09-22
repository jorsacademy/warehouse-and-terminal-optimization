from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp as pl


@dataclass(frozen=True)
class Container:
    container_id: int
    weight: int
    volume: int
    value: int
    priority: int
    hazardous: int
    reefer: int
    region: str


CONTAINERS: List[Container] = [
    Container(1, 18, 28, 5200, 3, 0, 0, "port"),
    Container(2, 22, 35, 6100, 2, 1, 0, "port"),
    Container(3, 16, 24, 4800, 1, 0, 1, "port"),
    Container(4, 28, 40, 7200, 3, 0, 0, "port"),
    Container(5, 24, 32, 6800, 2, 1, 0, "starboard"),
    Container(6, 20, 30, 5900, 1, 0, 1, "starboard"),
    Container(7, 26, 38, 7500, 3, 0, 0, "starboard"),
    Container(8, 14, 20, 4300, 1, 0, 1, "starboard"),
    Container(9, 30, 42, 8000, 2, 1, 0, "center"),
    Container(10, 17, 25, 5100, 2, 0, 0, "center"),
    Container(11, 21, 31, 6400, 3, 0, 0, "center"),
    Container(12, 25, 36, 7000, 1, 0, 1, "center"),
]

MAX_WEIGHT = 170
MAX_VOLUME = 245
MAX_REEFER = 3
MAX_HAZARDOUS = 2
MIN_PRIORITY_1 = 3
MAX_PORT_STARBOARD_IMBALANCE = 25
PRIORITY_BONUS = 250

INCOMPATIBLE_PAIRS: List[Tuple[int, int]] = [
    (2, 5),
    (5, 9),
    (2, 9),
    (3, 8),
]


def build_model() -> tuple[pl.LpProblem, Dict[int, pl.LpVariable]]:
    model = pl.LpProblem("Freighter_Container_Loading", pl.LpMaximize)
    x = {
        c.container_id: pl.LpVariable(
            f"load_{c.container_id}", lowBound=0, upBound=1, cat=pl.LpBinary
        )
        for c in CONTAINERS
    }

    model += pl.lpSum(
        c.value * x[c.container_id]
        + PRIORITY_BONUS * (4 - c.priority) * x[c.container_id]
        for c in CONTAINERS
    ), "Total_Economic_Value_With_Priority"

    model += pl.lpSum(c.weight * x[c.container_id] for c in CONTAINERS) <= MAX_WEIGHT, "Weight_Capacity"
    model += pl.lpSum(c.volume * x[c.container_id] for c in CONTAINERS) <= MAX_VOLUME, "Volume_Capacity"
    model += pl.lpSum(c.reefer * x[c.container_id] for c in CONTAINERS) <= MAX_REEFER, "Reefer_Capacity"
    model += pl.lpSum(c.hazardous * x[c.container_id] for c in CONTAINERS) <= MAX_HAZARDOUS, "Hazardous_Limit"
    model += pl.lpSum((1 if c.priority == 1 else 0) * x[c.container_id] for c in CONTAINERS) >= MIN_PRIORITY_1, "Priority_1_Minimum"

    for i, j in INCOMPATIBLE_PAIRS:
        model += x[i] + x[j] <= 1, f"Incompatible_{i}_{j}"

    port_weight = pl.lpSum(c.weight * x[c.container_id] for c in CONTAINERS if c.region == "port")
    starboard_weight = pl.lpSum(c.weight * x[c.container_id] for c in CONTAINERS if c.region == "starboard")

    model += port_weight - starboard_weight <= MAX_PORT_STARBOARD_IMBALANCE, "Balance_Port_Minus_Starboard"
    model += starboard_weight - port_weight <= MAX_PORT_STARBOARD_IMBALANCE, "Balance_Starboard_Minus_Port"

    return model, x


def solve_model(msg: bool = False) -> dict:
    model, x = build_model()
    solver = pl.PULP_CBC_CMD(msg=msg)
    status_code = model.solve(solver)
    status = pl.LpStatus[status_code]

    selected = [i for i, var in x.items() if pl.value(var) > 0.5]

    selected_containers = [c for c in CONTAINERS if c.container_id in selected]
    total_weight = sum(c.weight for c in selected_containers)
    total_volume = sum(c.volume for c in selected_containers)
    total_reefers = sum(c.reefer for c in selected_containers)
    total_hazardous = sum(c.hazardous for c in selected_containers)
    priority_1_count = sum(c.priority == 1 for c in selected_containers)
    port_weight = sum(c.weight for c in selected_containers if c.region == "port")
    starboard_weight = sum(c.weight for c in selected_containers if c.region == "starboard")

    return {
        "status": status,
        "objective": pl.value(model.objective),
        "selected": selected,
        "total_weight": total_weight,
        "total_volume": total_volume,
        "total_reefers": total_reefers,
        "total_hazardous": total_hazardous,
        "priority_1_count": priority_1_count,
        "port_weight": port_weight,
        "starboard_weight": starboard_weight,
        "imbalance": abs(port_weight - starboard_weight),
    }


def main() -> None:
    result = solve_model(msg=True)

    print(f"Status: {result['status']}")
    print(f"Objective value: {result['objective']:.0f}")
    print(f"Selected containers: {result['selected']}")
    print(f"Total weight: {result['total_weight']} / {MAX_WEIGHT}")
    print(f"Total volume: {result['total_volume']} / {MAX_VOLUME}")
    print(f"Reefer containers: {result['total_reefers']} / {MAX_REEFER}")
    print(f"Hazardous containers: {result['total_hazardous']} / {MAX_HAZARDOUS}")
    print(f"Priority-1 containers: {result['priority_1_count']} / minimum {MIN_PRIORITY_1}")
    print(f"Port weight: {result['port_weight']}")
    print(f"Starboard weight: {result['starboard_weight']}")
    print(f"Port-starboard imbalance: {result['imbalance']} / {MAX_PORT_STARBOARD_IMBALANCE}")


if __name__ == "__main__":
    main()
