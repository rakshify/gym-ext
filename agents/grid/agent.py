import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple

import numpy as np

from agents.base import ModelFreeAgent
from algorithms import Algorithm
from gym_ext.envs import Env


class GridAgent(ModelFreeAgent):
    name = "GridWorld"
    
    def get_action(self, state: int) -> int:
        q_val, action = self.get_qval_action(state)
        return action

    def get_qval_action(self, state: int) -> int:
        qvals = self.model.predict(state)
        return self.policy.get_action(qvals)

    def update_model(self, update: Any, step_size: float, state: int,
                     action: int):
        self.model.update(update * step_size, state, action)

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
