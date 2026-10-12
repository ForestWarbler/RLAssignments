"""Run every .in maze and organize the experiment outputs by case."""

from pathlib import Path
from shutil import copyfile

from experiment import export_data, format_summary, run_experiment, write_csv
from maze import Maze
from plot_results import render_results

GAMMA = 0.99
TOLERANCE = 1e-8
COMPARISON_GAMMAS = (0.9, 0.95, 0.99)
ROOT = Path(__file__).resolve().parent


def write_index(output_dir: Path, summary: list[dict]) -> None:
    """Create an index with baseline results and links to each case's artifacts."""
    lines = [
        "# Maze experiment results",
        "",
        "Each case has its own directory named after its source `.in` file.",
        "Run all cases again with `.venv/bin/python as_1/run_cases.py` from the project root.",
        "The script locates its input and output directories relative to itself.",
        "",
        f"Baseline: gamma = {GAMMA}, tolerance = {TOLERANCE:g}.",
        "Comparison gammas: " + ", ".join(map(str, COMPARISON_GAMMAS)) + ".",
        "All runs use synchronous value iteration with initial values of zero.",
        "Movement is deterministic; each action costs 1, including collisions",
        "and entry into the terminal goal. The goal value is zero.",
        "",
        "## Baseline results",
        "",
        "| Case | Grid | States | Iterations | Path steps | Start value | Verified | Outputs |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary:
        case = row["case"]
        links = (
            f"[Summary]({case}/summary.txt) · "
            f"[Path]({case}/path.png) · [JSON]({case}/results.json)"
        )
        lines.append(
            f"| {case} | {row['rows']} × {row['cols']} | {row['states']} | "
            f"{row['iterations']} | {row['path_length']} | {row['start_value']:.6f} | "
            f"{'PASS' if row['verified'] else 'FAIL'} | {links} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- [summary.csv](summary.csv): one baseline result per case.",
        "- [gamma_summary.csv](gamma_summary.csv): all case/gamma comparisons.",
        "- `maze.in`: a copy of the input grid used for that case.",
        "- `summary.txt` and `run.log`: the textual experiment report and policy.",
        "- `results.json`: parameters, values, policy, path, convergence, and verification.",
        "- `values.csv`, `path.csv`, `convergence.csv`, and `gamma_comparison.csv`: data tables.",
        "- `maze`, `policy`, `path`, `values`, `convergence`, and `gamma_comparison`:",
        "  figures saved as both PNG and PDF.",
        "",
        "Verification compares path length and every greedy action with BFS distances,",
        "checks every state value against its discounted shortest-path cost, and checks",
        "the start value against the rollout return.",
        "Solver runtimes measure one value-iteration run each; they are not benchmarks",
        "and exclude plotting, file export, and verification.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "results"
    summary = []
    gamma_summary = []

    for input_file in sorted((ROOT / "cases").glob("*.in")):
        case = input_file.stem
        maze = Maze.from_file(input_file)
        result = run_experiment(maze, GAMMA, TOLERANCE)
        result["case"] = case
        result["input_file"] = input_file.relative_to(ROOT).as_posix()
        runs = {
            gamma: result if gamma == GAMMA else run_experiment(maze, gamma, TOLERANCE)
            for gamma in COMPARISON_GAMMAS
        }
        result["gamma_comparison"] = [
            {
                "gamma": gamma,
                "iterations": run["iterations"],
                "path_length": run["path_length"],
                "start_value": run["start_value"],
                "runtime_seconds": run["runtime_seconds"],
                "verified": run["verified"],
            }
            for gamma, run in runs.items()
        ]

        case_dir = output_dir / case
        export_data(result, case_dir)
        copyfile(input_file, case_dir / "maze.in")
        (case_dir / "run.log").write_text(format_summary(result), encoding="utf-8")
        render_results(result, case_dir)

        summary.append({
            "case": case,
            "rows": len(maze.rows),
            "cols": len(maze.rows[0]),
            "states": len(maze.states),
            "gamma": GAMMA,
            "iterations": result["iterations"],
            "path_length": result["path_length"],
            "start_value": result["start_value"],
            "undiscounted_return": result["undiscounted_return"],
            "runtime_seconds": result["runtime_seconds"],
            "verified": result["verified"],
        })
        gamma_summary.extend({"case": case, **row} for row in result["gamma_comparison"])
        print(
            f"{case}: states={len(maze.states)}, iterations={result['iterations']}, "
            f"steps={result['path_length']}, verified={result['verified']}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "gamma_summary.csv", gamma_summary)
    write_index(output_dir, summary)


if __name__ == "__main__":
    main()
