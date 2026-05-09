import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Grid: TypeAlias = NDArray[np.bool_]
"""3D boolean array representing a grid map where True indicates passable cells."""

Coord: TypeAlias = tuple[int, int, int]
"""Coordinate tuple (z, y, x) representing a position in the grid."""

Action: TypeAlias = tuple[int, int, int]
"""Tuple indicating the deltas (d_z, d_y, d_x) of a movement between neighbouring nodes."""


@dataclass
class Config:
    positions: list[Coord] = field(default_factory=lambda: [])

    def __getitem__(self, k: int) -> Coord:
        return self.positions[k]

    def __setitem__(self, k: int, coord: Coord) -> None:
        self.positions[k] = coord

    def __len__(self) -> int:
        return len(self.positions)

    def __hash__(self) -> int:
        return hash(tuple(self.positions))
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another configuration.

        Args:
            other: Object to compare with.

        Returns:
            True if other is a Config with identical positions.
        """
        if not isinstance(other, Config):
            return NotImplemented
        return self.positions == other.positions

    def append(self, coord: Coord) -> None:
        self.positions.append(coord)

    def __iter__(self) -> Iterator[Coord]:
        """Iterate over agent positions.

        Returns:
            Iterator over agent positions.
        """
        return iter(self.positions)
    
    def count(self, val: Coord) -> int:
        """
        Args:
            val: Coord to count for.

        Returns:
            number of ocurrences of ```val``` Coord in this configuration.
        """
        return self.positions.count(val)

    def init_populate(self, coords: list[Coord]) -> None:
        """Adds a batch of new agent positions to this configuration.

        Args:
            coords: The list with (z, y, x) coordinate to add.
        """
        self.positions = coords

    def copy(self):
        return Config(self.positions)
        


Configs: TypeAlias = list[Config]


@dataclass
class Deadline:
    time_limit_ms: int

    def __post_init__(self) -> None:
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return (time.time() - self.start_time) * 1000

    @property
    def is_expired(self) -> bool:
        return self.elapsed > self.time_limit_ms


def get_grid(map_file: str | Path) -> Grid:
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str | Path, N: int | None = None) -> tuple[Config, Config]:
    with open(scen_file, "r") as f:
        starts, goals = Config(), Config()
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    z, y, x = coord
    if z < 0 or z >= grid.shape[0] or y < 0 or y >= grid.shape[1] or x < 0 or x >= grid.shape[2] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: z, y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    z, y, x = coord

    if x > 0: 
        if not z and grid[z, y, x - 1]:
            neigh.append((z, y, x - 1))
        # if grid[int(not z), y, x - 1]:      # vertical diagonal
        #     neigh.append((int(not z), y, x - 1))
        # horizontal diagonals
        if not z:
            if y > 0 and grid[z, y - 1, x - 1]:
                neigh.append((z, y - 1, x - 1))
            if y < grid.shape[1] - 1 and grid[z, y + 1, x - 1]:
                neigh.append((z, y + 1, x - 1))

    if x < grid.shape[2] - 1: 
        if not z and grid[z, y, x + 1]:
            neigh.append((z, y, x + 1))
        # if grid[int(not z), y, x + 1]:      # vertical diagonal
        #     neigh.append((int(not z), y, x + 1))
        # horizontal diagonals
        if not z:
            if y > 0 and grid[z, y - 1, x + 1]:
                neigh.append((z, y - 1, x + 1))
            if y < grid.shape[1] - 1 and grid[z, y + 1, x + 1]:
                neigh.append((z, y + 1, x + 1))

    if y > 0: 
        if not z and grid[z, y - 1, x]:
            neigh.append((z, y - 1, x))
        # if grid[int(not z), y - 1, x]:
        #     neigh.append((int(not z), y - 1, x))     # vertical diagonal

    if y < grid.shape[1] - 1:
        if not z and grid[z, y + 1, x]:
            neigh.append((z, y + 1, x))
    #     if grid[int(not z), y + 1, x]:
    #         neigh.append((int(not z), y + 1, x))     # vertical diagonal

    # if grid[int(not z), y, x]:
    #     neigh.append((int(not z), y, x))

    return neigh

def get_actions(coord: Coord) -> list[Action]:
    """Possible actions: up, right, down, left, stay, diagonals."""
    # z, _, _ = coord
    # z_op = 1 if not z else -1
    # actions = [(0, 0, 0), (z_op, 0, 0), (z_op, 0, 1), (z_op, -1, 0), (z_op, 1, 0), (z_op, 0, -1)]
    # if not z:
    #     actions += [(0, -1, 0), (0, 0, 1), (0, 1, 0), (0, 0, -1), (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)]  # d_z, d_y, d_x
    # actions = [(0, -1, 0), (0, 0, 1), (0, 1, 0), (0, 0, -1), (0, 0, 0)]     # 4-connectivity
    actions = [(0, -1, 0), (0, 0, 1), (0, 1, 0), (0, 0, -1), (0, 0, 0), (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)]     # 8-connectivity
    return actions

def calculate_action(v_to: Coord, v_from: Coord) -> Action:
    """Calculate (d_z, d_y, d_x) when moving from v_from to v_to"""
    return (v_to[0] - v_from[0], v_to[1] - v_from[1], v_to[2] - v_from[2])

def get_merging_actions(Q_from: Config, Q_to: Config) -> dict[int, Action]:
    """Calculate the corresponding action when an agent comes to the same vertex already occupied by another agent"""
    return {i: calculate_action(v_to, Q_from[i]) for i, v_to in enumerate(Q_to) if Q_to.count(v_to) == 2}


def save_configs_for_visualizer(configs: Configs, filename: str | Path) -> None:
    output_dirname = Path(filename).parent
    if not output_dirname.exists():
        output_dirname.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    assert len(solution) > 0, "invalid solution, empty"

    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False


def get_sum_of_loss(configs: Configs) -> int:
    cost = 0
    for t in range(1, len(configs)):
        cost += sum(
            [
                not (v_from == v_to == goal)
                for (v_from, v_to, goal) in zip(configs[t - 1], configs[t], configs[-1])
            ]
        )
    return cost
