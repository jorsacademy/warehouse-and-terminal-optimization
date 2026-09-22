import csv
from pathlib import Path

from src.guillotine import guillotine_best_area_fit
from src.metrics import packing_metrics, validate_packing
from src.models import Item
from src.shelf import best_fit_decreasing_shelf, first_fit_decreasing_shelf
from src.visualization import save_packing_plot


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sample_orders.csv"
OUTPUT_DIR = ROOT / "outputs"
BIN_WIDTH = 60
BIN_HEIGHT = 40


def load_items(path: Path) -> list[Item]:
    """Load the synthetic item dataset."""

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            Item(
                item_id=row["item_id"],
                width=int(row["width"]),
                height=int(row["height"]),
            )
            for row in reader
        ]


def print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Bins used: {metrics['bin_count']}")
    print(f"Packed area: {metrics['packed_area']}")
    print(f"Unused area: {metrics['unused_area']}")
    print(f"Average utilization: {metrics['average_utilization']:.2%}")


def main() -> None:
    items = load_items(DATA_FILE)

    algorithms = {
        "First-Fit Decreasing Shelf": first_fit_decreasing_shelf,
        "Best-Fit Decreasing Shelf": best_fit_decreasing_shelf,
        "Guillotine Best-Area-Fit": guillotine_best_area_fit,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, algorithm in algorithms.items():
        placements = algorithm(items, BIN_WIDTH, BIN_HEIGHT)
        validate_packing(placements, BIN_WIDTH, BIN_HEIGHT)
        metrics = packing_metrics(placements, BIN_WIDTH, BIN_HEIGHT)
        print_metrics(name, metrics)

        filename = name.lower().replace(" ", "_").replace("-", "_") + ".png"
        save_packing_plot(
            placements,
            BIN_WIDTH,
            BIN_HEIGHT,
            OUTPUT_DIR / filename,
            name,
        )


if __name__ == "__main__":
    main()
