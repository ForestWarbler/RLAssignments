"""Solve a deterministic maze MDP with synchronous value iteration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

State = tuple[int, int]
Move = tuple[int, int]
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Up, right, down, left.
DEFAULT_MAZE = (
    "########",
    "#......#",
    "S.##.#.#",
    "#..##..#",
    "##..#.##",
    "#.#.#..#",
    "#....#.G",
    "########",
)


class Maze:
    """Represent open cells as states and treat the goal as terminal."""

    def __init__(self, rows: Sequence[str] = DEFAULT_MAZE):
        self.rows = tuple(rows)
        self.states = tuple(
            (r, c)
            for r, row in enumerate(self.rows)
            for c, cell in enumerate(row)
            if cell != "#"
        )
        self._open = set(self.states)
        self.start = next((r, c) for r, c in self.states if self.rows[r][c] == "S")
        self.goal = next((r, c) for r, c in self.states if self.rows[r][c] == "G")

    @classmethod
    def from_file(cls, path: str | Path) -> "Maze":
        """Read a plain-text maze grid from a .in file."""
        return cls(Path(path).read_text(encoding="utf-8").splitlines())

    def step(self, state: State, move: Move) -> tuple[State, float, bool]:
        """Apply a (row, column) displacement and return the transition."""
        if state == self.goal:
            return self.goal, 0.0, True
        dr, dc = move
        successor = (state[0] + dr, state[1] + dc)
        if successor not in self._open:
            successor = state
        # Entering the goal and hitting a wall both incur the step cost.
        return successor, -1.0, successor == self.goal


@dataclass
class ValueIterationResult:
    """Store the converged value function, greedy policy, and update history."""

    values: dict[State, float]
    policy: dict[State, Move]
    deltas: tuple[float, ...]

    @property
    def iterations(self) -> int:
        return len(self.deltas)


def value_iteration(
    maze: Maze,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
) -> ValueIterationResult:
    """Apply synchronous Bellman optimality updates until values stabilize."""
    values = dict.fromkeys(maze.states, 0.0)
    deltas = []

    def action_value(state: State, move: Move) -> float:
        successor, reward, done = maze.step(state, move)
        return reward + (0.0 if done else gamma * values[successor])

    while True:
        # Read the old values throughout a sweep, then replace the whole table.
        updated = {
            state: max(action_value(state, move) for move in MOVES)
            for state in maze.states
        }
        deltas.append(max(abs(updated[state] - values[state]) for state in maze.states))
        values = updated
        if deltas[-1] < tolerance:
            break

    # The order in MOVES breaks ties consistently.
    policy = {
        state: max(MOVES, key=lambda move: action_value(state, move))
        for state in maze.states
        if state != maze.goal
    }
    return ValueIterationResult(values, policy, tuple(deltas))


@dataclass
class Trajectory:
    """Store a rollout, including both its starting and terminal states."""

    states: tuple[State, ...]
    actions: tuple[Move, ...]
    rewards: tuple[float, ...]

    @property
    def length(self) -> int:
        return len(self.actions)


def rollout(maze: Maze, policy: Mapping[State, Move]) -> Trajectory:
    """Follow the policy from the start to the goal."""
    state = maze.start
    states, actions, rewards = [state], [], []
    while state != maze.goal:
        move = policy[state]
        state, reward, _ = maze.step(state, move)
        states.append(state)
        actions.append(move)
        rewards.append(reward)
    return Trajectory(tuple(states), tuple(actions), tuple(rewards))
