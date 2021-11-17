"""Implements the continuous grid environment."""

import json
import os

from typing import Any, Dict, List, Tuple

# import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from gym_ext.envs.grid.grid_env import GridEnv
from gym_ext.envs.grid.utils import GridRead


class ContGridEnv(GridEnv):
    """
    Description:
        A rectangular continuos world without obstacles where the object moves
        around based on its velocity. The object starts from a random position
        and the goal is to reach any of the target states.
    Observation:
        Type: Box(4)
        Num     Observation               Min            Max
        0       Obj x Position            0              width of rect
        1       Obj y Position            0              height of rect
        2       Obj x Velocity            -1             1
        3       Obj y Velocity            -1             1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Push obj up
        1     Push obj down
        2     Push obj to the left
        3     Push obj to the right
        Note: The amount the velocity that is reduced or increased is fixed;
        The fix value is set in during initialization via action_effect field.
    Reward:
        Reward is -1 for every step taken and 0 for the termination step
    Starting State:
        Pos observations are assigned a uniform random value in [0..0.1]
        Vel observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Object is in the vicinity of any of the termination states.
        Vicinity is set during initialization via reach_threshold field.
    """

    name = "ContGridWorld"
    version = "v0"
    entry_point = "gym_ext.envs:ContGridEnv"

    def __init__(self, grid_size: Tuple[int, int] = None,
                 target_states: List[Tuple[int, int]] = None):
        """Initialize a ContGridEnv."""
        self.action_space = spaces.Discrete(4)
        self.action_effect = 0.2
        self.reach_threshold = 0.1
        self.steps_beyond_done = None
        self.elapsed_steps = None
        if grid_size is None:
            grid_size = (4, 4)
            # grid_size = self.read_grid_size()
        if target_states is None:
            target_states = [(1, 1), (2, 2)]
            # target_states = self.read_target_states(grid_size)
        self.seed()
        self.set_grid(grid_size, target_states)

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

        x_pos = self.state[0]
        y_pos = self.state[1]
        x_vel = self.state[2]
        y_vel = self.state[3]
        # ACTIONS = ["up", "down", "left", "right"]
        # print(f"Current position is: ({x_pos}, {y_pos}).")
        # print(f"Current velocity is: ({x_vel}, {y_vel}).")
        # print(f"Taking action {ACTIONS[action]}")
        minx = self.observation_space.low[0]
        miny = self.observation_space.low[1]
        maxx = self.observation_space.high[0]
        maxy = self.observation_space.high[1]
        nx_vel = x_vel
        ny_vel = y_vel

        if self.target_achieved(self.state[:2]):
            # print("Box is in vicinity of one of the targets1.")
            # input("+" * 80)
            state = self.state
        else:
            # Up action
            if action == 0:
                ny_vel -= self.action_effect
                ny_vel = max(ny_vel, miny - y_pos)
            # Down action
            elif action == 1:
                ny_vel += self.action_effect
                ny_vel = min(ny_vel, maxy - y_pos)
            # Up action
            elif action == 2:
                nx_vel -= self.action_effect
                nx_vel = max(nx_vel, minx - x_pos)
            # Down action
            else:
                nx_vel += self.action_effect
                nx_vel = min(nx_vel, maxx - x_pos)
            nx_pos = x_pos + nx_vel
            ny_pos = y_pos + ny_vel
            # print(f"New velocity is: ({nx_vel}, {ny_vel}).")
            # Can't go past boundaries
            nx_pos = min(maxx, max(minx, nx_pos))
            ny_pos = min(maxy, max(miny, ny_pos))
            # print(f"New position is: ({nx_pos}, {ny_pos}).")
            # print("+" * 80)
            # input("+" * 80)
            state = np.array([nx_pos, ny_pos, nx_vel, ny_vel])
            
        # # Action not allowed if results in blocked state
        # if self.states[nsi] == "x":
        #     nsi = si
        done = self.target_achieved(state[:2])
        if not done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            # print("Box is in vicinity of one of the targets.")
            # input("+" * 80)
            # Reached target state
            self.steps_beyond_done = 0
            reward = 0.0
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

        self.state = state
        self.elapsed_steps += 1

        return (self.state, reward, done, {})

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def target_achieved(self, pos: np.ndarray) -> bool:
        dist_from_target = np.array([
            np.linalg.norm(ts - pos) for ts in self.target_states
        ])
        return np.any(dist_from_target <= self.reach_threshold)

    def reset(self) -> int:
        """Start a new episode by sampling a new state."""
        # blocked or target states not allowed to be sampled
        while True:
            state = self.np_random.uniform(low=0, high=0.1, size=(4,))
            if not self.target_achieved(state[:2]):
                self.elapsed_steps = 0
                self.steps_beyond_done = None
                self.state = state
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

    def read_grid_size(self):
        """Read the grid size from CLI."""
        return GridRead.read_grid_size(1, 3)

    def read_target_states(self, grid_size: Tuple[int, int]):
        return GridRead.read_target_states(
            grid_size[0], grid_size[1], 1, 3, read_float=True)

    def set_grid(self, grid_size: Tuple[int, int],
                 target_states: List[Tuple[int, int]]):
        """
        Set the grid.

        Args:
            grid_size: Size of grid to set.
            target_states: target states of the environment.
        """
        # Set low and high limits of the box state.
        low = np.array(
            [
                0,
                0,
                -1,
                -1
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                grid_size[0],
                grid_size[1],
                1,
                1
            ],
            dtype=np.float32,
        )
        self.grid_size = grid_size
        self.target_states = target_states
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

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
        metadata["env"].update({
            "grid_size": os.path.join(model_dir, "grid_size.json"),
            "targets": os.path.join(model_dir, "targets.json")
        })
        with open(metadata["env"]["grid_size"], "w") as f:
            json.dump(self.grid_size, f)
        with open(metadata["env"]["targets"], "w") as f:
            json.dump(self.target_states, f)
        return metadata

    @classmethod
    def load_from_meta(cls, meta: Dict[str, Any]) -> "ContGridEnv":
        """
        Load a ContGridEnv from metadata.

        Args:
            meta: A dictionary of metadata.

        Returns:
            A ContGridEnv.
        """
        with open(meta["grid_size"]) as f:
            grid_size = tuple(json.load(f))
        with open(meta["targets"]) as f:
            targets = [tuple(s) for s in json.load(f)]
        env = cls(grid_size, targets)
        return env
