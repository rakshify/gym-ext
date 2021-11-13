import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple

import numpy as np

from solvers.grid.utils import GridRead
from gym_ext.envs import GridEnv


class GridSlover(Example):
    name = "GridWorld"

    @classmethod
    def train(cls, model_dir: str, **kwargs):
        grid_size = GridRead.read_grid_size(1, 3)
        grid = GridRead.read_grid(grid_size, 1, 3)
        env = GridEnv(grid)
        
        start = time.time()
        cls.solve(env)
        print("\n", "-" * 80)
        print("Agent learnt the q-vals to be:")
        print(agent.qvals)

        metadata = env.update_metadata({"model_dir": model_dir})
        metadata = agent.update_metadata(metadata)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def predict(cls, model_dir: str, **kwargs):
        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)
        env = load_env_from_meta(metadata)
        agent = load_agent_from_meta(metadata)

        path_grid = cls._get_path_grid(env.state_space.grid)
        print("The grid is: ")
        cls._print_path_grid(path_grid)
        row, col = len(env.state_space.grid), len(env.state_space.grid[0])
        env.set_start_state(*(GridRead.read_start_state(row, col, 1, 3)))
        start = time.time()
        state = env.start_state
        path_grid[state.row][state.col] = "o"
        cls._print_path_grid(path_grid)
        solution = []
        while not state.is_target():
            action = agent.get_action(state)
            path_grid[state.row][state.col] = action.action
            reward, state = env.play(state, action)
            path_grid[state.row][state.col] = "o"
            cls._print_path_grid(path_grid)

    @staticmethod
    def _print_path_grid(path_grid: List[List[str]]):
        col = len(path_grid[0])
        print("\n")
        print(".___" * col + ".")
        for row in path_grid:
            print("| " + " | ".join(row) + " |")
            print("|___" * col + "|")

    @staticmethod
    def _get_path_grid(grid: List[List[str]]) -> List[List[str]]:
        path_grid = []
        for row in grid:
            path_row = []
            for item in row:
                if item == "t":
                    path_row.append("*")
                elif item == "x":
                    path_row.append("x")
                else:
                    path_row.append("-")
            path_grid.append(path_row)
        return path_grid

    @classmethod
    def _read_wind(cls, grid_size: Tuple[int, int], **kwargs
                   ) -> Tuple[str, Dict[str, Any]]:
        choice = GridRead.is_windy(1, 3)
        env = cls.name
        if choice in ("y", "yes"):
            row, col = grid_size
            wind = []
            direction = GridRead.read_wind_direction(1, 3)
            if direction in ('u', 'up', 'd', 'down'):
                loop = col
                var = "col"
            else:
                loop = row
                var = "row"
            kwargs["wind_factors"] = GridRead.read_wind_factors(
                loop, var, 1, 3)
            kwargs["direction"] = direction
            env = f"windy_{env}"
        return env, kwargs
