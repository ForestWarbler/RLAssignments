"""Verify the maze model and value iteration against shortest-path results."""

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from maze import DEFAULT_MAZE, MOVES, Maze, rollout, value_iteration
from experiment import export_data, format_summary, run_experiment


class MazeTests(unittest.TestCase):
    def setUp(self):
        self.maze = Maze()

    def test_default_maze_has_expected_size_and_shortest_path(self):
        self.assertEqual(self.maze.rows, DEFAULT_MAZE)
        self.assertEqual(len(self.maze.states), 27)
        distances = self.maze.distances_to_goal()
        self.assertEqual(set(distances), set(self.maze.states))
        self.assertEqual(distances[self.maze.goal], 0)
        self.assertEqual(distances[self.maze.start], 15)

    def test_walls_and_boundaries_still_cost_one_step(self):
        for move in ((-1, 0), (1, 0), (0, -1)):
            with self.subTest(move=move):
                self.assertEqual(
                    self.maze.step(self.maze.start, move),
                    (self.maze.start, -1.0, False),
                )
        self.assertEqual(self.maze.step(self.maze.start, (0, 1)), ((2, 1), -1.0, False))

    def test_entering_goal_costs_one_step_and_terminates(self):
        maze = Maze(("SG",))
        self.assertEqual(maze.step(maze.start, (0, 1)), (maze.goal, -1.0, True))
        for move in MOVES:
            with self.subTest(move=move):
                self.assertEqual(maze.step(maze.goal, move), (maze.goal, 0.0, True))


class MazeFileTests(unittest.TestCase):
    def test_file_loader_reads_grid_coordinates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.in"
            path.write_text("#####\n#S.G#\n#####\n", encoding="utf-8")
            maze = Maze.from_file(path)
        self.assertEqual(maze.rows, ("#####", "#S.G#", "#####"))
        self.assertEqual(maze.start, (1, 1))
        self.assertEqual(maze.goal, (1, 3))

    def test_file_cases_match_shortest_path_values(self):
        cases = Path(__file__).parent / "cases"
        self.assertEqual(Maze.from_file(cases / "assignment.in").rows, DEFAULT_MAZE)
        for path in sorted(cases.glob("*.in")):
            with self.subTest(case=path.name):
                maze = Maze.from_file(path)
                self.assertEqual(len(set(map(len, maze.rows))), 1)
                self.assertEqual(set(maze.distances_to_goal()), set(maze.states))
                for gamma in (0.9, 0.95, 0.99):
                    result = run_experiment(maze, gamma=gamma, tolerance=1e-8)
                    self.assertTrue(result["verified"], (path.name, gamma))


class ValueIterationTests(unittest.TestCase):
    def test_all_values_match_discounted_shortest_path_costs(self):
        maze = Maze()
        distances = maze.distances_to_goal()
        for gamma in (0.9, 0.95, 0.99):
            with self.subTest(gamma=gamma):
                result = value_iteration(maze, gamma=gamma)
                self.assertEqual(set(result.values), set(maze.states))
                self.assertEqual(set(result.policy), set(maze.states) - {maze.goal})
                self.assertEqual(result.values[maze.goal], 0.0)
                self.assertEqual(result.iterations, len(result.deltas))
                self.assertGreater(result.iterations, 0)
                self.assertLess(result.deltas[-1], 1e-8)
                self.assertTrue(all(math.isfinite(delta) and delta >= 0 for delta in result.deltas))
                for state, distance in distances.items():
                    expected = -sum(gamma**step for step in range(distance))
                    self.assertAlmostEqual(result.values[state], expected, places=7)
                    if state != maze.goal:
                        successor, _, _ = maze.step(state, result.policy[state])
                        self.assertEqual(distances[successor], distance - 1)

    def test_single_move_has_no_extra_goal_reward(self):
        maze = Maze(("SG",))
        result = value_iteration(maze)
        self.assertEqual(result.values[maze.start], -1.0)
        self.assertEqual(result.policy[maze.start], (0, 1))
        trajectory = rollout(maze, result.policy)
        self.assertEqual(tuple(trajectory.states), (maze.start, maze.goal))
        self.assertEqual(tuple(trajectory.actions), ((0, 1),))
        self.assertEqual(tuple(trajectory.rewards), (-1.0,))
        self.assertEqual(trajectory.length, 1)

    def test_equal_quality_actions_follow_declared_order(self):
        cases = (
            (("..G", ".S.", "..."), (-1, 0)),
            (("...", ".S.", "..G"), (0, 1)),
            (("...", ".S.", "G.."), (1, 0)),
        )
        for rows, expected_move in cases:
            with self.subTest(rows=rows):
                maze = Maze(rows)
                result = value_iteration(maze)
                self.assertEqual(result.policy[maze.start], expected_move)


class ExperimentTests(unittest.TestCase):
    def test_loose_tolerance_cannot_verify_a_policy_with_remote_loops(self):
        # The start can reach the goal even while distant states are unresolved.
        maze = Maze(("SG....",))
        result = run_experiment(maze, gamma=0.99, tolerance=0.99)
        self.assertEqual(result["path_length"], 1)
        self.assertFalse(result["verified"])

    def test_exported_displacements_preserve_policy_and_path(self):
        result = run_experiment(Maze(), gamma=0.99, tolerance=1e-8)
        result["gamma_comparison"] = [{
            "gamma": 0.99,
            "iterations": result["iterations"],
            "path_length": result["path_length"],
            "start_value": result["start_value"],
        }]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            export_data(result, output)
            saved = json.loads((output / "results.json").read_text())
            self.assertEqual(saved["path_actions"], result["path_actions"])
            self.assertEqual(saved["states"], result["states"])
            self.assertEqual(format_summary(saved), format_summary(result))
            with (output / "path.csv").open(newline="") as stream:
                path = list(csv.DictReader(stream))
            with (output / "values.csv").open(newline="") as stream:
                values = list(csv.DictReader(stream))

        for step, successor in zip(path, path[1:]):
            self.assertEqual(
                (int(step["row"]) + int(step["dr"]), int(step["col"]) + int(step["dc"])),
                (int(successor["row"]), int(successor["col"])),
            )
        self.assertEqual((path[-1]["dr"], path[-1]["dc"], path[-1]["reward"]), ("", "", ""))
        for row, state in zip(values, result["states"]):
            expected = state["action"]
            if expected is None:
                self.assertEqual((row["dr"], row["dc"]), ("", ""))
            else:
                self.assertEqual([int(row["dr"]), int(row["dc"])], expected)


class RolloutTests(unittest.TestCase):
    def test_default_policy_reaches_goal_along_a_shortest_path(self):
        maze = Maze()
        result = value_iteration(maze)
        trajectory = rollout(maze, result.policy)
        self.assertEqual(trajectory.states[0], maze.start)
        self.assertEqual(trajectory.states[-1], maze.goal)
        self.assertEqual(trajectory.length, 15)
        self.assertEqual(len(trajectory.states), trajectory.length + 1)
        self.assertEqual(len(trajectory.actions), trajectory.length)
        self.assertEqual(len(trajectory.rewards), trajectory.length)
        self.assertEqual(sum(trajectory.rewards), -15.0)
        self.assertEqual(len(set(trajectory.states)), len(trajectory.states))
        for index, (state, action, reward) in enumerate(
            zip(trajectory.states, trajectory.actions, trajectory.rewards)
        ):
            successor, expected_reward, done = maze.step(state, action)
            self.assertEqual(successor, trajectory.states[index + 1])
            self.assertEqual(reward, expected_reward)
            self.assertEqual(done, index == trajectory.length - 1)

if __name__ == "__main__":
    unittest.main()
