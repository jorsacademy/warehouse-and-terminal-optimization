import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exhaustive_validator import exhaustive_optimum
from freighter_optimization import (
    INCOMPATIBLE_PAIRS,
    MAX_HAZARDOUS,
    MAX_PORT_STARBOARD_IMBALANCE,
    MAX_REEFER,
    MAX_VOLUME,
    MAX_WEIGHT,
    MIN_PRIORITY_1,
    solve_model,
)


def test_reference_optimum_matches_exhaustive_search():
    milp = solve_model(msg=False)
    brute = exhaustive_optimum()

    assert milp["status"] == "Optimal"
    assert milp["objective"] == brute["objective"] == 51900
    assert milp["selected"] == brute["selected"] == [1, 4, 5, 6, 8, 10, 11, 12]


def test_solution_satisfies_all_global_constraints():
    result = solve_model(msg=False)

    assert result["total_weight"] <= MAX_WEIGHT
    assert result["total_volume"] <= MAX_VOLUME
    assert result["total_reefers"] <= MAX_REEFER
    assert result["total_hazardous"] <= MAX_HAZARDOUS
    assert result["priority_1_count"] >= MIN_PRIORITY_1
    assert result["imbalance"] <= MAX_PORT_STARBOARD_IMBALANCE


def test_selected_containers_respect_incompatibilities():
    result = solve_model(msg=False)
    selected = set(result["selected"])

    for i, j in INCOMPATIBLE_PAIRS:
        assert not ({i, j} <= selected)
