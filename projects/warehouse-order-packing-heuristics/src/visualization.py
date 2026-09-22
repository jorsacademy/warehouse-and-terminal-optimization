from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .models import Placement


def save_packing_plot(
    placements: List[Placement],
    bin_width: int,
    bin_height: int,
    output_path: str | Path,
    title: str,
) -> None:
    """Save one image containing all used bins for a packing result."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bin_indices = sorted({placement.bin_index for placement in placements})
    if not bin_indices:
        return

    fig_width = max(6, 5 * len(bin_indices))
    fig, axes = plt.subplots(1, len(bin_indices), figsize=(fig_width, 5), squeeze=False)

    for column, bin_index in enumerate(bin_indices):
        ax = axes[0][column]
        ax.set_xlim(0, bin_width)
        ax.set_ylim(0, bin_height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Bin {bin_index + 1}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

        for placement in placements:
            if placement.bin_index != bin_index:
                continue

            rectangle = Rectangle(
                (placement.x, placement.y),
                placement.width,
                placement.height,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(rectangle)
            ax.text(
                placement.x + placement.width / 2,
                placement.y + placement.height / 2,
                placement.item_id,
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
