"""Render compact, serif figures inspired by the supplied SudokuRAG paper."""

from pathlib import Path

import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation

matplotlib.use("Agg")  # write files only; never open a GUI window
import matplotlib.pyplot as plt  # noqa: E402

# Match the blue, coral, and gray used in the paper's diagrams.
BLUE = "#479FEC"
CORAL = "#EE7058"
WALL_COLOR = "#D3D3D3"
FLOOR_COLOR = "#FFFFFF"
START_COLOR = "#DDEDFB"
GOAL_COLOR = "#F6D7CE"
PATH_COLOR = CORAL
TEXT_COLOR = "#171717"
ACCENT_COLOR = BLUE
MUTED_COLOR = "#555555"
GRID_COLOR = "#AFAFAF"

CELL_COLORS = {"#": WALL_COLOR, "S": START_COLOR, "G": GOAL_COLOR}
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
    "font.weight": "bold",
    "mathtext.fontset": "stix",
    "mathtext.default": "bf",
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TEXT_COLOR,
    "axes.axisbelow": True,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "grid.linestyle": ":",
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.labelspacing": 0.3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def maze_figure(maze, title, corner_labels=False):
    """Draw the maze in the solver's row/column coordinates; return (fig, ax)."""
    rows, cols = len(maze), len(maze[0])
    grid_width = max(2.8, cols * 0.42)
    grid_height = grid_width * rows / cols
    width, height = grid_width + 0.60, grid_height + 1.20
    fig, ax = plt.subplots(figsize=(width, height))
    # Keep cells readable and reserve fixed physical space for labels and legends.
    fig.subplots_adjust(
        left=0.46 / width, right=1 - 0.14 / width,
        bottom=0.85 / height, top=1 - 0.35 / height,
    )
    for row, line in enumerate(maze):
        for col, cell in enumerate(line):
            ax.add_patch(
                Rectangle(
                    (col, row), 1, 1, facecolor=CELL_COLORS.get(cell, FLOOR_COLOR),
                    edgecolor="#888888", linewidth=0.65, zorder=0,
                )
            )
            if cell in "SG":
                offset = 0.13 if corner_labels else 0.5
                ax.text(
                    col + offset, row + offset, cell,
                    ha="left" if corner_labels else "center",
                    va="top" if corner_labels else "center",
                    fontsize=7 if corner_labels else 10,
                    fontweight="bold", color=TEXT_COLOR, zorder=5,
                )

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([col + 0.5 for col in range(cols)], range(cols))
    ax.set_yticks([row + 0.5 for row in range(rows)], range(rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
    ax.set_title(title, pad=7)
    return fig, ax


def add_legend(ax):
    """Explain the maze colors below the axes."""
    legend_transform = ax.transAxes + ScaledTranslation(
        0, -0.48, ax.figure.dpi_scale_trans
    )
    ax.legend(
        handles=[
            Patch(facecolor=WALL_COLOR, edgecolor="#888888", linewidth=0.65, label="Wall"),
            Patch(facecolor=FLOOR_COLOR, edgecolor="#888888", linewidth=0.65, label="Corridor"),
            Patch(facecolor=START_COLOR, edgecolor="#888888", linewidth=0.65, label="Start (S)"),
            Patch(facecolor=GOAL_COLOR, edgecolor="#888888", linewidth=0.65, label="Goal (G)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=legend_transform,
        ncol=4, frameon=False, fontsize=7,
        handlelength=0.9, handletextpad=0.35, columnspacing=0.75,
    )


def plot_maze(result):
    fig, ax = maze_figure(result["maze"], "Maze environment")
    add_legend(ax)
    return fig


def plot_policy(result):
    gamma = result["parameters"]["gamma"]
    fig, ax = maze_figure(
        result["maze"], rf"Optimal policy ($\gamma={gamma:g}$)", corner_labels=True
    )
    for state in result["states"]:
        if state["action"] is None:
            continue
        dr, dc = state["action"]
        x, y = state["col"] + 0.5, state["row"] + 0.5
        ax.annotate(
            "", xy=(x + 0.28 * dc, y + 0.28 * dr),
            xytext=(x - 0.24 * dc, y - 0.24 * dr),
            arrowprops={
                "arrowstyle": "-|>", "color": BLUE,
                "lw": 1.6, "mutation_scale": 10,
            },
            zorder=3,
        )
    add_legend(ax)
    return fig


def plot_path(result):
    path = result["path"]
    steps = "step" if result["path_length"] == 1 else "steps"
    fig, ax = maze_figure(
        result["maze"], f"Optimal path ({result['path_length']} {steps})", corner_labels=True
    )
    ax.plot(
        [col + 0.5 for _, col in path], [row + 0.5 for row, _ in path],
        color=PATH_COLOR, linewidth=2.2, marker="o", markersize=4.1,
        markerfacecolor="white", markeredgewidth=1.1, zorder=2,
    )
    for index, (row, col) in enumerate(path):
        ax.text(
            col + 0.76, row + 0.80, str(index), color=TEXT_COLOR,
            fontsize=7, ha="center", va="center", zorder=4,
        )
    add_legend(ax)
    fig.text(
        0.54, 0.065 / fig.get_figheight(), "Numbers indicate time steps, starting at 0.",
        ha="center", fontsize=7, color=MUTED_COLOR,
    )
    return fig


def plot_values(result):
    maze = result["maze"]
    gamma = result["parameters"]["gamma"]
    fig, ax = maze_figure(
        maze, rf"Optimal state values ($\gamma={gamma:g}$)", corner_labels=True
    )
    values = [state["value"] for state in result["states"]]
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap("Reds")
    for state in result["states"]:
        row, col, value = state["row"], state["col"], state["value"]
        color = cmap(norm(value))
        text_color = "white" if norm(value) > 0.60 else TEXT_COLOR
        ax.add_patch(
            Rectangle(
                (col, row), 1, 1, facecolor=color,
                edgecolor="#888888", linewidth=0.65, zorder=1,
            )
        )
        ax.text(
            col + 0.5, row + 0.55, f"{value:.2f}", ha="center",
            va="center", fontsize=6.6, color=text_color, zorder=3,
        )
        if maze[row][col] in "SG":
            # Cover the base label with a color matched to the value cell.
            ax.text(
                col + 0.13, row + 0.13, maze[row][col],
                ha="left", va="top", fontsize=6.5, fontweight="bold",
                color=text_color, zorder=6,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 0.1},
            )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.035
    )
    colorbar.set_label(r"State value $V^*(s)$", labelpad=4)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.outline.set_linewidth(0.9)
    colorbar.ax.tick_params(width=0.9, length=2.5)
    return fig


def plot_convergence(result):
    fig, ax = plt.subplots(figsize=(3.65, 2.6), layout="constrained")
    iterations = [entry["iteration"] for entry in result["convergence"]]
    deltas = [entry["max_delta"] for entry in result["convergence"]]
    tolerance = result["parameters"]["tolerance"]
    coefficient, _, exponent = f"{tolerance:g}".partition("e")
    tolerance_label = coefficient
    if exponent:
        power = rf"10^{{{int(exponent)}}}"
        tolerance_label = power if coefficient == "1" else rf"{coefficient}\times {power}"
    # A log axis cannot show exact zero, so lift zeros to a floor below the rest.
    floor = min([tolerance, *(d for d in deltas if d > 0)]) / 10
    ax.semilogy(
        iterations, [max(d, floor) for d in deltas],
        color=PATH_COLOR, marker="o", markersize=3.8, linewidth=1.8,
        markerfacecolor="white", markeredgecolor=TEXT_COLOR, markeredgewidth=0.8,
        label=r"Maximum update $\Delta_k$",
    )
    ax.axhline(
        tolerance, color=ACCENT_COLOR, linestyle="--", linewidth=1.5,
        label=rf"Tolerance $\varepsilon={tolerance_label}$",
    )
    # Iteration stops at the first delta below tolerance, so only the last can be 0.
    if deltas[-1] == 0:
        ax.annotate(
            "Exact zero\n(shown below tolerance)",
            xy=(iterations[-1], floor),
            xytext=(0.95, 0.68) if len(iterations) == 2 else (-12, 26),
            textcoords="axes fraction" if len(iterations) == 2 else "offset points",
            ha="right",
            fontsize=7, color=TEXT_COLOR,
            arrowprops={"arrowstyle": "-", "color": MUTED_COLOR, "lw": 0.9},
        )
    ax.set_title("Value iteration convergence", pad=7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Maximum update $\Delta_k$ (log scale)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="y")
    ax.legend(
        frameon=True, facecolor="white", edgecolor="none", framealpha=1,
        loc="center left", bbox_to_anchor=(0, 0.35 if len(iterations) == 2 else 0.5),
        fontsize=7.5,
    )
    ax.margins(x=0.04, y=0.12)
    return fig


def plot_gamma_comparison(result):
    rows = result["gamma_comparison"]
    positions = range(len(rows))
    start_values = [row["start_value"] for row in rows]
    iteration_counts = [row["iterations"] for row in rows]
    path_lengths = [row["path_length"] for row in rows]

    fig, (value_ax, iteration_ax) = plt.subplots(
        1, 2, figsize=(6.5, 2.65), layout="constrained"
    )

    value_ax.bar(
        positions, start_values, color=CORAL, edgecolor=TEXT_COLOR,
        linewidth=0.8, width=0.55, zorder=3,
    )
    value_ax.set_title("(a) Start-state value", pad=7)
    value_ax.set_ylabel(r"Optimal return $V^*(S)$")
    value_ax.set_ylim(min(start_values) * 1.24, 0)
    for x, value in zip(positions, start_values):
        value_ax.annotate(
            f"{value:.4f}", xy=(x, value), xytext=(0, -7),
            textcoords="offset points", ha="center", va="top", fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )

    iteration_ax.bar(
        positions, iteration_counts, color=BLUE, edgecolor=TEXT_COLOR,
        linewidth=0.8, width=0.55, zorder=3,
    )
    iteration_ax.set_title("(b) Iterations to convergence", pad=7)
    iteration_ax.set_ylabel("Update sweeps")
    iteration_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    iteration_ax.set_ylim(0, max(iteration_counts) * 1.22)
    for x, count in zip(positions, iteration_counts):
        iteration_ax.annotate(
            str(count), xy=(x, count), xytext=(0, 5),
            textcoords="offset points", ha="center", fontsize=7.5,
        )

    for ax in (value_ax, iteration_ax):
        ax.set_xticks(positions, [f"{row['gamma']:g}" for row in rows])
        ax.set_xlabel(r"Discount factor $\gamma$")
        ax.grid(axis="y", zorder=0)

    if len(set(path_lengths)) == 1:
        caption = f"All settings yield a {path_lengths[0]}-step optimal path."
    else:
        caption = "Path lengths: " + "; ".join(
            f"gamma {row['gamma']:g}: {row['path_length']} steps" for row in rows
        ) + "."
    if len(set(iteration_counts)) == 1:
        caption += f" Each converges in {iteration_counts[0]} sweeps."
    fig.supxlabel(caption, fontsize=7.5, color=MUTED_COLOR)
    return fig


FIGURES = {
    "maze": plot_maze,
    "policy": plot_policy,
    "path": plot_path,
    "values": plot_values,
    "convergence": plot_convergence,
    "gamma_comparison": plot_gamma_comparison,
}


def render_results(result: dict, output_dir: Path) -> list[Path]:
    """Save every figure as PNG and PDF; return the saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with plt.rc_context(STYLE):
        for name, draw in FIGURES.items():
            fig = draw(result)
            for extension in ("png", "pdf"):
                path = output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                saved.append(path)
            plt.close(fig)
    return saved
