from collections import defaultdict
from typing import Dict, List

from .models import Placement


def packing_metrics(
    placements: List[Placement], bin_width: int, bin_height: int
) -> Dict[str, float | int]:
    """Calculate summary metrics for a packing result."""

    if not placements:
        return {
            "packed_area": 0,
            "bin_count": 0,
            "average_utilization": 0.0,
            "unused_area": 0,
        }

    bin_area = bin_width * bin_height
    packed_by_bin = defaultdict(int)

    for placement in placements:
        packed_by_bin[placement.bin_index] += placement.area

    bin_count = len(packed_by_bin)
    packed_area = sum(packed_by_bin.values())
    total_capacity = bin_count * bin_area

    return {
        "packed_area": packed_area,
        "bin_count": bin_count,
        "average_utilization": packed_area / total_capacity,
        "unused_area": total_capacity - packed_area,
    }


def rectangles_overlap(a: Placement, b: Placement) -> bool:
    """Return True when two placements overlap inside the same bin."""

    if a.bin_index != b.bin_index:
        return False

    return not (
        a.x + a.width <= b.x
        or b.x + b.width <= a.x
        or a.y + a.height <= b.y
        or b.y + b.height <= a.y
    )


def validate_packing(
    placements: List[Placement], bin_width: int, bin_height: int
) -> None:
    """Raise ValueError when a packing is outside bounds or contains overlaps."""

    for placement in placements:
        if placement.x < 0 or placement.y < 0:
            raise ValueError(f"{placement.item_id}: negative placement coordinate.")
        if placement.x + placement.width > bin_width:
            raise ValueError(f"{placement.item_id}: exceeds bin width.")
        if placement.y + placement.height > bin_height:
            raise ValueError(f"{placement.item_id}: exceeds bin height.")

    for index, first in enumerate(placements):
        for second in placements[index + 1 :]:
            if rectangles_overlap(first, second):
                raise ValueError(
                    f"Overlap detected between {first.item_id} and {second.item_id} "
                    f"in bin {first.bin_index}."
                )
