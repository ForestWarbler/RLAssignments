"""Run and export the maze value-iteration experiment."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform
from time import perf_counter

from maze import MOVES, Maze, rollout, value_iteration


def run_experiment(maze: Maze, gamma: float, tolerance: float) -> dict:
    """Solve the MDP and record the values, policy, and rollout."""
    started = perf_counter()
    solution = value_iteration(maze, gamma, tolerance)
    runtime = perf_counter() - started
    trajectory = rollout(maze, solution.policy)
    discounted_return = math.fsum(
        gamma**t * reward for t, reward in enumerate(trajectory.rewards)
    )

    return {
        "maze": list(maze.rows),
        "parameters": {
            "gamma": gamma,
            "tolerance": tolerance,
            "initial_value": 0.0,
            "action_order": [list(move) for move in MOVES],
            "transition_model": "deterministic",
            "collision_rule": "stay in place and receive -1",
            "step_reward": -1.0,
            "terminal_value": 0.0,
            "update_rule": "synchronous",
        },
        "python_version": platform.python_version(),
        "state_count": len(maze.states),
        "start": list(maze.start),
        "goal": list(maze.goal),
        "iterations": solution.iterations,
        "runtime_seconds": runtime,
        "convergence": [
            {"iteration": i, "max_delta": delta}
            for i, delta in enumerate(solution.deltas, start=1)
        ],
        "states": [
            {
                "row": state[0],
                "col": state[1],
                "value": solution.values[state],
                "action": list(solution.policy[state]) if state != maze.goal else None,
            }
            for state in maze.states
        ],
        "path": [list(state) for state in trajectory.states],
        "path_actions": [list(move) for move in trajectory.actions],
        "path_rewards": list(trajectory.rewards),
        "path_length": trajectory.length,
        "undiscounted_return": math.fsum(trajectory.rewards),
        "discounted_return": discounted_return,
        "start_value": solution.values[maze.start],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write records with a stable header order and portable line endings."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_summary(result: dict) -> str:
    """Create a concise summary suitable for the terminal and report notes."""
    policy = {(s["row"], s["col"]): s["action"] for s in result["states"]}
    arrows = {(-1, 0): "^", (0, 1): ">", (1, 0): "v", (0, -1): "<"}
    policy_rows = [
        "".join(
            cell if cell in "#SG" else arrows[tuple(policy[r, c])]
            for c, cell in enumerate(row)
        )
        for r, row in enumerate(result["maze"])
    ]
    lines = [
        "Maze MDP - Value Iteration",
        f"States (including goal): {result['state_count']}",
        f"Gamma: {result['parameters']['gamma']}",
        f"Tolerance: {result['parameters']['tolerance']:.1e}",
        f"Iterations: {result['iterations']}",
        f"Final maximum update: {result['convergence'][-1]['max_delta']:.10g}",
        f"Solver runtime: {result['runtime_seconds']:.6f} s (one run)",
        f"Path length: {result['path_length']} steps",
        f"Undiscounted return: {result['undiscounted_return']:.0f}",
        f"Discounted return: {result['discounted_return']:.10f}",
        f"Start value: {result['start_value']:.10f}",
        "", "Greedy policy (S=start, G=goal):", *policy_rows,
        "", "Path actions: " + " ".join(arrows[tuple(move)] for move in result["path_actions"]),
        "", "Gamma comparison:",
        "gamma    iterations    path_steps    start_value",
    ]
    lines += [
        f"{row['gamma']:<8g} {row['iterations']:<13d} "
        f"{row['path_length']:<13d} {row['start_value']:.10f}"
        for row in result["gamma_comparison"]
    ]
    return "\n".join(lines) + "\n"


def export_data(result: dict, output_dir: Path) -> None:
    """Save machine-readable results and a human-readable experiment summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "convergence.csv", result["convergence"])
    write_csv(output_dir / "values.csv", [
        {
            "row": state["row"],
            "col": state["col"],
            "value": state["value"],
            "dr": state["action"][0] if state["action"] is not None else "",
            "dc": state["action"][1] if state["action"] is not None else "",
        }
        for state in result["states"]
    ])
    write_csv(output_dir / "gamma_comparison.csv", result["gamma_comparison"])
    # The terminal state has no move or reward.
    steps = zip(
        result["path"],
        [*result["path_actions"], ["", ""]],
        [*result["path_rewards"], ""],
    )
    write_csv(output_dir / "path.csv", [
        {"step": i, "row": row, "col": col, "dr": dr, "dc": dc, "reward": reward}
        for i, ((row, col), (dr, dc), reward) in enumerate(steps)
    ])
    (output_dir / "summary.txt").write_text(format_summary(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--compare-gammas", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    parser.add_argument("--maze-file", type=Path, help="A .in grid containing #, ., S, and G.")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: results/<case>)."
    )
    parser.add_argument("--no-plots", action="store_true", help="Run without Matplotlib.")
    args = parser.parse_args()
    case = args.maze_file.stem if args.maze_file else "assignment"
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results" / case
    maze = Maze.from_file(args.maze_file) if args.maze_file else Maze()
    result = run_experiment(maze, args.gamma, args.tolerance)
    result["case"] = case
    runs = {
        gamma: result if gamma == args.gamma else run_experiment(
            maze, gamma, args.tolerance
        )
        for gamma in args.compare_gammas
    }
    result["gamma_comparison"] = [
        {
            "gamma": gamma,
            "iterations": run["iterations"],
            "path_length": run["path_length"],
            "start_value": run["start_value"],
            "runtime_seconds": run["runtime_seconds"],
        }
        for gamma, run in runs.items()
    ]
    export_data(result, output_dir)

    print(format_summary(result))
    if not args.no_plots:
        from plot_results import render_results

        print(f"Saved {len(render_results(result, output_dir))} figure files.")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
