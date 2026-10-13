"""Compare value-iteration stopping counts on the largest supplied maze."""

import argparse
import csv
import json
from pathlib import Path

from maze import Maze, value_iteration
from plot_results import BLUE, CORAL, STYLE, TEXT_COLOR, plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
DEFAULT_GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)


def check_policy(maze, policy):
    """Stop at the goal or a repeated state, even for a poor greedy policy."""
    state = maze.start
    seen = set()
    path = [state]
    while state != maze.goal and state not in seen:
        seen.add(state)
        state, _, _ = maze.step(state, policy[state])
        path.append(state)
    return state == maze.goal, path


def draw_comparison(records, output_dir):
    """Render the measured counts, distinguishing successful and cyclic policies."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.3, 3.6))
        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.98)
        positions = range(len(records))
        counts = [row['iterations'] for row in records]
        bars = ax.bar(
            positions, counts, width=1.0,
            color=[BLUE if row['policy_reaches_goal'] else CORAL for row in records],
            edgecolor=TEXT_COLOR, linewidth=0.8, zorder=3,
        )
        ax.bar_label(bars, padding=5, fontsize=13)
        ax.set_xticks(positions, [f"{row['gamma']:g}" for row in records])
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel(r'Discount factor $\gamma$', labelpad=8, fontsize=14)
        ax.set_ylabel('Iterations to stopping criterion', labelpad=9, fontsize=14)
        ax.set_xlim(-0.5, len(records) - 0.5)
        ax.set_ylim(0, max(counts) * 1.30)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis='y')
        ax.legend(handles=[
            Patch(facecolor=CORAL, edgecolor=TEXT_COLOR, label='Failed'),
            Patch(facecolor=BLUE, edgecolor=TEXT_COLOR, label='Success'),
        ], loc='upper left', fontsize=13, frameon=True,
            facecolor='white', edgecolor='none', framealpha=1)
        for extension in ('png', 'pdf'):
            fig.savefig(output_dir / f'gamma_iterations.{extension}', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze-file', type=Path, help='Defaults to the case with the most states.')
    parser.add_argument('--gammas', type=float, nargs='+', default=DEFAULT_GAMMAS)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--output-dir', type=Path, help='Defaults to results/<case>/gamma_sweep.')
    args = parser.parse_args()
    if not 0 < args.tolerance < float('inf'):
        parser.error('--tolerance must be positive and finite')
    if any(not 0 < gamma < 1 for gamma in args.gammas):
        parser.error('each gamma must be strictly between 0 and 1')

    input_file = args.maze_file or max(
        sorted((ROOT / 'cases').glob('*.in')),
        key=lambda path: len(Maze.from_file(path).states),
    )
    maze = Maze.from_file(input_file)
    case = input_file.stem
    output_dir = args.output_dir or ROOT / 'results' / case / 'gamma_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)
    records, runs = [], []

    for gamma in sorted(set(args.gammas)):
        solution = value_iteration(maze, gamma=gamma, tolerance=args.tolerance)
        reaches_goal, path = check_policy(maze, solution.policy)
        row = {
            'gamma': gamma,
            'tolerance': args.tolerance,
            'iterations': solution.iterations,
            'final_max_delta': solution.deltas[-1],
            'start_value': solution.values[maze.start],
            'policy_reaches_goal': reaches_goal,
            'path_length': len(path) - 1 if reaches_goal else None,
            'steps_before_repeat': None if reaches_goal else len(path) - 1,
        }
        records.append(row)
        runs.append({**row, 'deltas': list(solution.deltas), 'policy_path': path})
        outcome = f'goal in {len(path) - 1} steps' if reaches_goal else 'repeated state'
        print(f'gamma={gamma:g}: {solution.iterations} iterations; {outcome}')

    with (output_dir / 'gamma_iterations.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    payload = {
        'case': case,
        'maze': list(maze.rows),
        'state_count': len(maze.states),
        'initial_value': 0.0,
        'update_rule': 'synchronous',
        'stopping_rule': 'max_s |V_new(s) - V_old(s)| < tolerance',
        'runs': runs,
    }
    (output_dir / 'gamma_iterations.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8',
    )
    draw_comparison(records, output_dir)
    print(f'Charts and data: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
