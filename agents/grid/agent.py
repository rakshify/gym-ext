import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from agents.base import ModelFreeAgent
from algorithms import Algorithm
from gym_ext.envs import Env


class GridAgent(ModelFreeAgent):
    name = "GridWorld"
    
    def get_action(self, state: int) -> int:
        q_val, action = self.get_qval_action(state)
        return action

    def get_qval_action(self, state: Union[int, np.ndarray]) -> int:
        qvals = self.model.predict(state)
        return self.policy.get_action(qvals)

    def update_model(self, update: Any):
        self.model.update(update)

    def q_grad(self, state: Union[int, np.ndarray], action: int) -> np.ndarray:
        return self.model.grad(state, action)

    def explore_policy(self):
        self.policy.explore()

    @property
    def vec_shape(self):
        return self.model.vec_shape

    @property
    def n_features(self):
        return self.env.observation_space.n

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
