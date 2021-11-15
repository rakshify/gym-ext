"""This implements greedy policy."""

from typing import List, Tuple

import numpy as np

from policies.policy import Policy


class Greedy(Policy):
    """This implements greedy policy."""
    name = 'greedy'
    
    def __init__(self, epsilon: float = 0.0):
        self.epsilon = epsilon

    def get_action(self, values: List[float]) -> Tuple[float, int]:
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(values))
        else:
            action = np.argmax(values)
        return values[action], action

    def update_policy(self, update: float, **kwargs):
        self.epsilon = update

    def serialize(self):
        return {
            "name": self.name,
            "epsilon": self.epsilon
        }
