import pytest

from src.metrics import packing_metrics, validate_packing
from src.models import Item
from src.shelf import best_fit_decreasing_shelf, first_fit_decreasing_shelf


BIN_WIDTH = 60
BIN_HEIGHT = 40


@pytest.fixture
def sample_items() -> list[Item]:
    return [
        Item("A", 30, 15),
        Item("B", 20, 12),
        Item("C", 15, 10),
        Item("D", 25, 8),
        Item("E", 10, 6),
    ]


def test_first_fit_produces_valid_packing(sample_items: list[Item]) -> None:
    placements = first_fit_decreasing_shelf(sample_items, BIN_WIDTH, BIN_HEIGHT)
    validate_packing(placements, BIN_WIDTH, BIN_HEIGHT)
    assert len(placements) == len(sample_items)


def test_best_fit_produces_valid_packing(sample_items: list[Item]) -> None:
    placements = best_fit_decreasing_shelf(sample_items, BIN_WIDTH, BIN_HEIGHT)
    validate_packing(placements, BIN_WIDTH, BIN_HEIGHT)
    assert len(placements) == len(sample_items)


def test_oversized_item_is_rejected() -> None:
    with pytest.raises(ValueError):
        first_fit_decreasing_shelf([Item("X", 61, 10)], BIN_WIDTH, BIN_HEIGHT)


def test_metrics_match_total_item_area(sample_items: list[Item]) -> None:
    placements = first_fit_decreasing_shelf(sample_items, BIN_WIDTH, BIN_HEIGHT)
    metrics = packing_metrics(placements, BIN_WIDTH, BIN_HEIGHT)
    expected_area = sum(item.area for item in sample_items)
    assert metrics["packed_area"] == expected_area
