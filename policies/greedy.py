"""This implements greedy policy."""

from typing import Any, Dict, List, Tuple

import numpy as np

from policies.policy import Policy


class Greedy(Policy):
    """This implements greedy policy."""
    name = 'greedy'
    
    def __init__(self, epsilon: float = 0.99, decay: float = 0.99):
        self.epsilon = epsilon
        self.decay = decay

    def get_action(self, values: List[float]) -> Tuple[float, int]:
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(values))
        else:
            action = np.argmax(values)
        return values[action], action

    def exploit(self, **kwargs):
        self.epsilon *= self.decay

    def explore(self, **kwargs):
        self.epsilon = min(0.99, self.epsilon / self.decay)

    def serialize(self):
        return {
            "name": self.name,
            "epsilon": self.epsilon
        }

    def load_vars(self, vars: Dict[str, Any]):
        self.epsilon = vars["epsilon"]
