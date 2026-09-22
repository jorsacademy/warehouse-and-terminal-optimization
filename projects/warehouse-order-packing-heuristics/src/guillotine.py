from dataclasses import dataclass
from typing import List

from .models import Item, Placement
from .shelf import _validate_items


@dataclass(frozen=True)
class _FreeRectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


def _split_free_rectangle(space: _FreeRectangle, item: Item) -> List[_FreeRectangle]:
    """Split the remaining L-shaped space into two non-overlapping rectangles."""

    right = _FreeRectangle(
        x=space.x + item.width,
        y=space.y,
        width=space.width - item.width,
        height=item.height,
    )
    top = _FreeRectangle(
        x=space.x,
        y=space.y + item.height,
        width=space.width,
        height=space.height - item.height,
    )

    return [rect for rect in (right, top) if rect.width > 0 and rect.height > 0]


def guillotine_best_area_fit(
    items: List[Item], bin_width: int, bin_height: int
) -> List[Placement]:
    """Pack items with a Guillotine Best-Area-Fit heuristic.

    Items are sorted by decreasing area. For each item, the algorithm selects the
    feasible free rectangle that leaves the least unused area after placement.
    Rotation is not allowed.
    """

    _validate_items(items, bin_width, bin_height)
    ordered_items = sorted(items, key=lambda item: (item.area, item.height, item.width), reverse=True)

    free_spaces: List[List[_FreeRectangle]] = []
    placements: List[Placement] = []

    for item in ordered_items:
        best_choice = None

        for bin_index, spaces in enumerate(free_spaces):
            for space_index, space in enumerate(spaces):
                if item.width <= space.width and item.height <= space.height:
                    waste = space.area - item.area
                    short_side = min(space.width - item.width, space.height - item.height)
                    score = (waste, short_side, bin_index, space_index)
                    if best_choice is None or score < best_choice[0]:
                        best_choice = (score, bin_index, space_index, space)

        if best_choice is None:
            bin_index = len(free_spaces)
            initial_space = _FreeRectangle(0, 0, bin_width, bin_height)
            free_spaces.append([])
            chosen_space = initial_space
        else:
            _, bin_index, space_index, chosen_space = best_choice
            del free_spaces[bin_index][space_index]

        placements.append(
            Placement(
                item_id=item.item_id,
                bin_index=bin_index,
                x=chosen_space.x,
                y=chosen_space.y,
                width=item.width,
                height=item.height,
            )
        )

        free_spaces[bin_index].extend(_split_free_rectangle(chosen_space, item))

    return placements
