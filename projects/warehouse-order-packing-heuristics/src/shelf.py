from dataclasses import dataclass
from typing import Callable, Iterable, List

from .models import Item, Placement


@dataclass
class _Shelf:
    y: int
    height: int
    used_width: int = 0


def _validate_items(items: Iterable[Item], bin_width: int, bin_height: int) -> None:
    if bin_width <= 0 or bin_height <= 0:
        raise ValueError("Bin dimensions must be positive.")

    for item in items:
        if item.width <= 0 or item.height <= 0:
            raise ValueError(f"{item.item_id}: dimensions must be positive.")
        if item.width > bin_width or item.height > bin_height:
            raise ValueError(
                f"{item.item_id}: {item.width}x{item.height} does not fit inside "
                f"a {bin_width}x{bin_height} bin."
            )


def _pack_shelf(
    items: List[Item],
    bin_width: int,
    bin_height: int,
    shelf_selector: Callable[[List[_Shelf], Item, int], int | None],
) -> List[Placement]:
    _validate_items(items, bin_width, bin_height)
    ordered_items = sorted(items, key=lambda item: (item.height, item.width), reverse=True)

    bins: List[List[_Shelf]] = []
    placements: List[Placement] = []

    for item in ordered_items:
        placed = False

        for bin_index, shelves in enumerate(bins):
            shelf_index = shelf_selector(shelves, item, bin_width)
            if shelf_index is None:
                continue

            shelf = shelves[shelf_index]
            placements.append(
                Placement(
                    item_id=item.item_id,
                    bin_index=bin_index,
                    x=shelf.used_width,
                    y=shelf.y,
                    width=item.width,
                    height=item.height,
                )
            )
            shelf.used_width += item.width
            placed = True
            break

        if placed:
            continue

        for bin_index, shelves in enumerate(bins):
            used_height = sum(shelf.height for shelf in shelves)
            if used_height + item.height <= bin_height:
                shelves.append(_Shelf(y=used_height, height=item.height, used_width=item.width))
                placements.append(
                    Placement(
                        item_id=item.item_id,
                        bin_index=bin_index,
                        x=0,
                        y=used_height,
                        width=item.width,
                        height=item.height,
                    )
                )
                placed = True
                break

        if placed:
            continue

        bins.append([_Shelf(y=0, height=item.height, used_width=item.width)])
        placements.append(
            Placement(
                item_id=item.item_id,
                bin_index=len(bins) - 1,
                x=0,
                y=0,
                width=item.width,
                height=item.height,
            )
        )

    return placements


def _first_fit_selector(shelves: List[_Shelf], item: Item, bin_width: int) -> int | None:
    for index, shelf in enumerate(shelves):
        if item.height <= shelf.height and shelf.used_width + item.width <= bin_width:
            return index
    return None


def _best_fit_selector(shelves: List[_Shelf], item: Item, bin_width: int) -> int | None:
    candidates = []
    for index, shelf in enumerate(shelves):
        remaining_width = bin_width - shelf.used_width
        if item.height <= shelf.height and item.width <= remaining_width:
            candidates.append((remaining_width - item.width, shelf.height - item.height, index))

    if not candidates:
        return None

    return min(candidates)[2]


def first_fit_decreasing_shelf(
    items: List[Item], bin_width: int, bin_height: int
) -> List[Placement]:
    """Pack items using a First-Fit Decreasing Height shelf heuristic."""

    return _pack_shelf(items, bin_width, bin_height, _first_fit_selector)


def best_fit_decreasing_shelf(
    items: List[Item], bin_width: int, bin_height: int
) -> List[Placement]:
    """Pack items using a Best-Fit Decreasing Height shelf heuristic."""

    return _pack_shelf(items, bin_width, bin_height, _best_fit_selector)
