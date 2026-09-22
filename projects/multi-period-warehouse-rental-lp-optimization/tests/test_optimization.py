import pytest

from warehouse_optimization import (
    MONTHLY_REQUIREMENTS,
    RENTAL_COSTS_USD_PER_SQM,
    monthly_covered_capacity,
    solve_model,
    validate_solution,
)


def test_optimal_solution_and_cost():
    result = solve_model()
    validate_solution(result)

    assert result.status == "Optimal"
    assert result.total_cost_usd == pytest.approx(7_650_000.0)

    plan = {
        (decision.start_month, decision.duration_months): decision.area_sqm
        for decision in result.decisions
    }
    assert plan == {
        (1, 5): pytest.approx(30_000.0),
        (3, 1): pytest.approx(10_000.0),
        (5, 1): pytest.approx(20_000.0),
    }


def test_monthly_requirements_are_covered():
    result = solve_model()
    coverage = monthly_covered_capacity(result.decisions)

    for month, requirement in MONTHLY_REQUIREMENTS.items():
        assert coverage[month] >= requirement


def test_cost_recalculation_matches_objective():
    result = solve_model()
    recomputed = sum(
        RENTAL_COSTS_USD_PER_SQM[d.duration_months] * d.area_sqm
        for d in result.decisions
    )
    assert recomputed == pytest.approx(result.total_cost_usd)
