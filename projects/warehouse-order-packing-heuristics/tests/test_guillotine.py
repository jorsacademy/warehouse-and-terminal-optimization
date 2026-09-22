import pytest

from src.guillotine import guillotine_best_area_fit
from src.metrics import packing_metrics, validate_packing
from src.models import Item


BIN_WIDTH = 60
BIN_HEIGHT = 40


@pytest.fixture
def sample_items() -> list[Item]:
    return [
        Item("A", 28, 16),
        Item("B", 24, 12),
        Item("C", 18, 10),
        Item("D", 15, 9),
        Item("E", 12, 7),
        Item("F", 20, 11),
    ]


def test_guillotine_produces_valid_packing(sample_items: list[Item]) -> None:
    placements = guillotine_best_area_fit(sample_items, BIN_WIDTH, BIN_HEIGHT)
    validate_packing(placements, BIN_WIDTH, BIN_HEIGHT)
    assert len(placements) == len(sample_items)


def test_guillotine_rejects_oversized_item() -> None:
    with pytest.raises(ValueError):
        guillotine_best_area_fit([Item("X", 20, 41)], BIN_WIDTH, BIN_HEIGHT)


def test_guillotine_metrics_match_item_area(sample_items: list[Item]) -> None:
    placements = guillotine_best_area_fit(sample_items, BIN_WIDTH, BIN_HEIGHT)
    metrics = packing_metrics(placements, BIN_WIDTH, BIN_HEIGHT)
    assert metrics["packed_area"] == sum(item.area for item in sample_items)
