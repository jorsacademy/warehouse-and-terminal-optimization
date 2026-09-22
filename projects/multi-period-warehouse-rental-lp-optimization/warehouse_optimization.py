"""Multi-period warehouse rental optimization using linear programming.

The model chooses how many square meters to rent in each month and for how
many months, while ensuring that every month's required warehouse capacity is
covered at minimum total cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pulp


MONTHLY_REQUIREMENTS: Dict[int, int] = {
    1: 30_000,
    2: 20_000,
    3: 40_000,
    4: 10_000,
    5: 50_000,
}

RENTAL_COSTS_USD_PER_SQM: Dict[int, int] = {
    1: 65,
    2: 100,
    3: 135,
    4: 160,
    5: 190,
}


@dataclass(frozen=True)
class RentalDecision:
    start_month: int
    duration_months: int
    area_sqm: float


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    total_cost_usd: float
    decisions: List[RentalDecision]


def build_model() -> Tuple[pulp.LpProblem, Dict[Tuple[int, int], pulp.LpVariable]]:
    """Build and return the warehouse-rental linear programming model."""
    months = sorted(MONTHLY_REQUIREMENTS)
    horizon = len(months)

    model = pulp.LpProblem("Multi_Period_Warehouse_Rental", pulp.LpMinimize)

    rental: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for start_month in months:
        max_duration = horizon - start_month + 1
        for duration in range(1, max_duration + 1):
            rental[(start_month, duration)] = pulp.LpVariable(
                f"rent_m{start_month}_d{duration}", lowBound=0, cat="Continuous"
            )

    model += pulp.lpSum(
        RENTAL_COSTS_USD_PER_SQM[duration] * variable
        for (start_month, duration), variable in rental.items()
    ), "Total_Rental_Cost_USD"

    for month, requirement in MONTHLY_REQUIREMENTS.items():
        active_capacity = pulp.lpSum(
            variable
            for (start_month, duration), variable in rental.items()
            if start_month <= month <= start_month + duration - 1
        )
        model += active_capacity >= requirement, f"Capacity_Month_{month}"

    return model, rental


def solve_model(msg: bool = False) -> OptimizationResult:
    """Solve the model and return a structured optimal solution."""
    model, rental = build_model()
    solver = pulp.PULP_CBC_CMD(msg=msg)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Optimization did not reach an optimal solution: {status}")

    decisions = [
        RentalDecision(start_month, duration, float(variable.value()))
        for (start_month, duration), variable in sorted(rental.items())
        if variable.value() is not None and variable.value() > 1e-9
    ]

    total_cost = float(pulp.value(model.objective))
    return OptimizationResult(status=status, total_cost_usd=total_cost, decisions=decisions)


def monthly_covered_capacity(decisions: List[RentalDecision]) -> Dict[int, float]:
    """Calculate total rented capacity available in each month."""
    coverage = {month: 0.0 for month in MONTHLY_REQUIREMENTS}
    for decision in decisions:
        for month in coverage:
            if decision.start_month <= month <= decision.start_month + decision.duration_months - 1:
                coverage[month] += decision.area_sqm
    return coverage


def validate_solution(result: OptimizationResult) -> None:
    """Validate feasibility and objective consistency of a solved result."""
    coverage = monthly_covered_capacity(result.decisions)

    for month, requirement in MONTHLY_REQUIREMENTS.items():
        if coverage[month] + 1e-9 < requirement:
            raise AssertionError(
                f"Month {month} is under-covered: {coverage[month]} < {requirement}"
            )

    recomputed_cost = sum(
        RENTAL_COSTS_USD_PER_SQM[d.duration_months] * d.area_sqm
        for d in result.decisions
    )
    if abs(recomputed_cost - result.total_cost_usd) > 1e-6:
        raise AssertionError(
            f"Objective mismatch: {recomputed_cost} != {result.total_cost_usd}"
        )


def main() -> None:
    result = solve_model()
    validate_solution(result)

    print("Optimal Rental Plan")
    print("-------------------")
    for decision in result.decisions:
        month_word = "month" if decision.duration_months == 1 else "months"
        print(
            f"Rent {decision.area_sqm:,.0f} m² in month {decision.start_month} "
            f"for {decision.duration_months} {month_word}."
        )

    print(f"\nMinimum Total Cost: ${result.total_cost_usd:,.2f}")


if __name__ == "__main__":
    main()
