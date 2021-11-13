"""Implements the grid environment."""

import os

from typing import Any, Dict, List, Tuple

import gym
from gym import spaces
import numpy as np


DEFAULT_GRID = [
    ["t", "o", "o", "o"],
    ["o", "o", "o", "o"],
    ["o", "o", "o", "o"],
    ["o", "o", "o", "t"]
]


class GridEnv(gym.Env):
    """Implements the grid environment."""

    name = "GridWorld"
    version = "v0"
    entry_point = "gym_ext.envs:GridEnv"

    def __init__(self, grid: List[List[str]] = DEFAULT_GRID):
        """Initialize a GridEnv."""
        self.action_space = spaces.Discrete(4)
        self.steps_beyond_done = None
        self.elapsed_steps = None
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

        return (self.state, reward, done, {})

    def reset(self) -> int:
        """Start a new episode by sampling a new state."""
        self.state = self.observation_space.sample()
        self.elapsed_steps = 0
        return self.state

    def set_grid(self, grid: List[List[str]]):
        """
        Set the grid.

        Args:
            grid: Grid to set.
        """
        self.states = np.array(grid).flatten()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.observation_space = spaces.Discrete(self.rows * self.cols)

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
        meta = {
            "grid": os.path.join(model_dir, "grid.py")
        }
        with open(meta["grid"], "wb") as f:
            np.save(f, self.states)

        metadata["env"].update(meta)
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
        env = cls()
        with open(meta["grid"], "rb") as f:
            env.states = np.load(f)
        return env
