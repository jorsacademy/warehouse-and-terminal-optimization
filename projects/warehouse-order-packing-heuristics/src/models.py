from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """A rectangular item to be packed."""

    item_id: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Placement:
    """A placed item inside a bin."""

    item_id: str
    bin_index: int
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height
