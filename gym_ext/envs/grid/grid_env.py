"""Implements the grid environment."""

import json
import os

from typing import Any, Dict, List, Tuple

# import gym
from gym import spaces
import numpy as np

from gym_ext.envs.base_env import Env
from gym_ext.envs.grid.utils import GridRead


class GridEnv(Env):
    """Implements the grid environment."""

    name = "GridWorld"
    version = "v0"
    entry_point = "gym_ext.envs:GridEnv"

    def __init__(self, grid: List[List[str]] = None):
        """Initialize a GridEnv."""
        self.action_space = spaces.Discrete(4)
        self.steps_beyond_done = None
        self.elapsed_steps = None
        if grid is None:
            grid = self.read_grid()
        self.set_grid(grid)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Play an action in a state.

        Args:
            action: Action to take.

        Returns:
            A tuple of (next state, reward, status, extra info).
        """
        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        err_msg = "Cannot call env.step() before calling reset()"
        assert self.elapsed_steps is not None, err_msg

        si = self.state
        nS = self.observation_space.n
        # nA = self.action_space.n

        # Up action
        if action == 0:
            nsi = si if si < self.cols else si - self.cols
        # Down action
        elif action == 1:
            nsi = si if si >= nS - self.cols else si + self.cols
        # Up action
        elif action == 2:
            nsi = si if si % self.cols == 0 else si - 1
        # Down action
        else:
            nsi = si if (si + 1) % self.cols == 0 else si + 1
        # Action not allowed if results in blocked state
        if self.states[nsi] == "x":
            nsi = si

        done = self.states[nsi] == "t"
        if not done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            # Reached target state
            self.steps_beyond_done = 0
            reward = -1.0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.state = nsi
        self.elapsed_steps += 1

        # For rendering
        ACTIONS = ["^", "v", "<", ">"]
        self.path_grid[si // self.cols][si % self.cols] = ACTIONS[action]
        self.path_grid[nsi // self.cols][nsi % self.cols] = "o"

        return (self.state, reward, done, {})

    def reset(self) -> int:
        """Start a new episode by sampling a new state."""
        self.state = self.observation_space.sample()
        self.elapsed_steps = 0
        return self.state

    def render(self, mode: str = "human") -> None:
        """
        Render the environment.

        Args:
            mode: Rendering mode.
        """
        if mode == "human":
            print("\n")
            print(".___" * self.cols + ".")
            for row in self.path_grid:
                print("| " + " | ".join(row) + " |")
                print("|___" * self.cols + "|")

    def read_grid(self):
        """Read the grid from CLI."""
        grid_size = GridRead.read_grid_size(1, 3)
        grid = GridRead.read_grid(grid_size, 1, 3)
        return grid

    def set_grid(self, grid: List[List[str]]):
        """
        Set the grid.

        Args:
            grid: Grid to set.
        """
        self.grid = grid
        self.states = np.array(grid).flatten()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.reset_path_grid()
        self.observation_space = spaces.Discrete(self.rows * self.cols)

    def reset_path_grid(self):
        """Reset the path grid."""
        self.path_grid = []
        for row in range(self.rows):
            prow = []
            for col in range(self.cols):
                si = row * self.cols + col
                if self.states[si] == "t":
                    prow.append("*")
                elif self.states[si] == "x":
                    prow.append("#")
                else:
                    prow.append("-")
            self.path_grid.append(prow)

    def update_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the metadata.

        Args:
            metadata: A dictionary of metadata.

        Returns:
            A dictionary of updated metadata.
        """
        metadata = super(GridEnv, self).update_metadata(metadata)
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        metadata["env"]["grid"] = os.path.join(model_dir, "grid.json")
        with open(metadata["env"]["grid"], "w") as f:
            json.dump(self.grid, f)
        return metadata

    @classmethod
    def load_from_meta(cls, meta: Dict[str, Any]) -> "GridEnv":
        """
        Load a GridEnv from metadata.

        Args:
            meta: A dictionary of metadata.

        Returns:
            A GridEnv.
        """
        with open(meta["grid"]) as f:
            grid = json.load(f)
        env = cls(grid)
        return env
