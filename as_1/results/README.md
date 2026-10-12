# Maze experiment results

Each case has its own directory named after its source `.in` file.
Run all cases again with `.venv/bin/python as_1/run_cases.py` from the project root.
The script locates its input and output directories relative to itself.

Baseline: gamma = 0.99, tolerance = 1e-08.
Comparison gammas: 0.9, 0.95, 0.99.
All runs use synchronous value iteration with initial values of zero.
Movement is deterministic; each action costs 1, including collisions
and entry into the terminal goal. The goal value is zero.

## Baseline results

| Case | Grid | States | Iterations | Path steps | Start value | Verified | Outputs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| adjacent | 3 × 4 | 2 | 2 | 1 | -1.000000 | PASS | [Summary](adjacent/summary.txt) · [Path](adjacent/path.png) · [JSON](adjacent/results.json) |
| assignment | 8 × 8 | 27 | 24 | 15 | -13.994165 | PASS | [Summary](assignment/summary.txt) · [Path](assignment/path.png) · [JSON](assignment/results.json) |
| corridor | 3 × 11 | 9 | 9 | 8 | -7.725531 | PASS | [Summary](corridor/summary.txt) · [Path](corridor/path.png) · [JSON](corridor/results.json) |
| dead_ends | 9 × 13 | 47 | 21 | 16 | -14.854223 | PASS | [Summary](dead_ends/summary.txt) · [Path](dead_ends/path.png) · [JSON](dead_ends/results.json) |
| detour | 7 × 9 | 31 | 15 | 14 | -13.125419 | PASS | [Summary](detour/summary.txt) · [Path](detour/path.png) · [JSON](detour/results.json) |
| larger | 15 × 15 | 102 | 47 | 40 | -33.102824 | PASS | [Summary](larger/summary.txt) · [Path](larger/path.png) · [JSON](larger/results.json) |
| loop | 7 × 9 | 20 | 11 | 10 | -9.561792 | PASS | [Summary](loop/summary.txt) · [Path](loop/path.png) · [JSON](loop/results.json) |
| open_room | 6 × 9 | 28 | 10 | 9 | -8.648275 | PASS | [Summary](open_room/summary.txt) · [Path](open_room/path.png) · [JSON](open_room/results.json) |

## Files

- [summary.csv](summary.csv): one baseline result per case.
- [gamma_summary.csv](gamma_summary.csv): all case/gamma comparisons.
- `maze.in`: a copy of the input grid used for that case.
- `summary.txt` and `run.log`: the textual experiment report and policy.
- `results.json`: parameters, values, policy, path, convergence, and verification.
- `values.csv`, `path.csv`, `convergence.csv`, and `gamma_comparison.csv`: data tables.
- `maze`, `policy`, `path`, `values`, `convergence`, and `gamma_comparison`:
  figures saved as both PNG and PDF.

Verification compares path length and every greedy action with BFS distances,
checks every state value against its discounted shortest-path cost, and checks
the start value against the rollout return.
Solver runtimes measure one value-iteration run each; they are not benchmarks
and exclude plotting, file export, and verification.
