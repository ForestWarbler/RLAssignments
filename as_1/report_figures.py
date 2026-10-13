"""Make caption-free path, policy, and value panels for the LaTeX report."""

import argparse
import json
import math
from pathlib import Path

from plot_results import (
    MaxNLocator,
    STYLE,
    ScalarMappable,
    draw_maze_cells,
    draw_path_cells,
    draw_policy_arrows,
    draw_value_cells,
    plt,
)


CASES = (
    "assignment", "complex_large", "adjacent", "corridor", "open_room",
    "detour", "dead_ends", "loop", "larger",
)
ROOT = Path(__file__).resolve().parent


def coordinate_ticks(size):
    """Keep the coordinate labels legible at the report's printed width."""
    step = max(1, math.ceil((size - 1) / 7))
    return sorted(set(range(0, size, step)) | {size - 1})


def format_grid(ax, maze, *, show_row_label=False):
    rows, cols = len(maze), len(maze[0])
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    columns, rows = coordinate_ticks(cols), coordinate_ticks(rows)
    ax.set_xticks([col + 0.5 for col in columns], columns)
    ax.set_yticks([row + 0.5 for row in rows], rows)
    ax.set_xlabel("Column", fontsize=8, labelpad=3)
    if show_row_label:
        ax.set_ylabel("Row", fontsize=8, labelpad=3)
    ax.tick_params(length=0, pad=2, labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def plot_triptych(result):
    """Use the actual 15.8 cm report width so external labels do not shrink."""
    maze = result["maze"]
    rows, cols = len(maze), len(maze[0])
    width = 15.8 / 2.54
    left, bottom, gap = 0.31, 0.36, 0.23
    colorbar_gap, colorbar_width, right = 0.075, 0.075, 0.42
    grid_width = (width - left - 2 * gap - colorbar_gap - colorbar_width - right) / 3
    grid_height = grid_width * rows / cols
    height = bottom + grid_height + 0.055
    cell_points = grid_width * 72 / cols
    value_font = min(9, 0.35 * cell_points)
    endpoint_font = min(9, 0.38 * cell_points)
    grid_linewidth = min(0.4, 0.023 * cell_points)
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([
        (left + index * (grid_width + gap)) / width,
        bottom / height, grid_width / width, grid_height / height,
    ]) for index in range(3)]
    for index, ax in enumerate(axes):
        draw_maze_cells(
            ax, maze, label_size=endpoint_font,
            grid_linewidth=grid_linewidth, draw_endpoints=index == 1,
        )
        format_grid(ax, maze, show_row_label=index == 0)

    path_ax, policy_ax, value_ax = axes
    draw_path_cells(
        path_ax, result, label_size=endpoint_font, grid_linewidth=grid_linewidth,
    )
    draw_policy_arrows(policy_ax, result)
    norm, cmap = draw_value_cells(
        value_ax, result, font_size=value_font, grid_linewidth=grid_linewidth,
    )

    # A separate colorbar axis preserves equal-sized maze grids in all panels.
    colorbar_ax = fig.add_axes([
        (left + 3 * grid_width + 2 * gap + colorbar_gap) / width,
        bottom / height, colorbar_width / width, grid_height / height,
    ])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar.set_label(r"$V^*(s)$", fontsize=8, labelpad=3)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(width=0.6, length=2, labelsize=7, pad=2)
    colorbar.locator = MaxNLocator(nbins=5, integer=True)
    colorbar.update_ticks()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report" / "figures")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for case in args.cases:
            result = json.loads((args.results_dir / case / "results.json").read_text())
            fig = plot_triptych(result)
            for extension in ("pdf", "png"):
                destination = args.output_dir / f"{case}_triptych.{extension}"
                fig.savefig(destination, dpi=args.dpi)
                print(destination)
            plt.close(fig)


if __name__ == "__main__":
    main()
