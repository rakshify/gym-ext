"""Implements the one-hot grid environment."""

from typing import Tuple

import numpy as np

from gym_ext.envs.grid.grid_env import GridEnv


class OneHotGridEnv(GridEnv):
    """Implements the one-hot grid environment."""

    name = "OneHotGrid"
    version = "v0"
    entry_point = "gym_ext.envs:OneHotGridEnv"

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

        si = np.argwhere(self.state == 1).flatten()[0]
        nS = self.observation_space.n
        # nA = self.action_space.n

        if self.states[si] == "t":
            nsi = si
        # Up action
        elif action == 0:
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

        self.state = np.zeros(self.observation_space.n)
        self.state[nsi] = 1
        self.elapsed_steps += 1

        # For rendering
        ACTIONS = ["^", "v", "<", ">"]
        self.path_grid[si // self.cols][si % self.cols] = ACTIONS[action]
        self.path_grid[nsi // self.cols][nsi % self.cols] = "o"

        return (self.state, reward, done, {})

    def reset(self) -> int:
        """Start a new episode by sampling a new state."""
        # blocked or target states not allowed to be sampled
        allowed_states = [i for i, st in enumerate(self.states)
                          if st not in ("x", "t")]
        si = np.random.choice(allowed_states)
        self.state = np.zeros(self.observation_space.n)
        self.state[si] = 1
        self.elapsed_steps = 0
        return self.state
