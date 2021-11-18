"""This implements greedy policy."""

from typing import Any, Dict, List, Tuple

import numpy as np

from policies.policy import Policy


class Greedy(Policy):
    """This implements greedy policy."""
    name = 'greedy'
    
    def __init__(self, epsilon: float = 0.99, decay: float = 0.99, min_eps: float = 0.01):
        self.epsilon = epsilon
        self.decay = decay
        self.min_eps = min_eps

    def get_action(self, values: List[float]) -> Tuple[float, int]:
        if np.random.rand() > self.epsilon:
            action = np.argmax(values)
        else:
            action = np.random.choice(len(values))
        return values, action

    def exploit(self, **kwargs):
        self.epsilon *= self.decay
        self.epsilon = max(self.min_eps, self.epsilon)

    def explore(self, **kwargs):
        self.epsilon = min(0.99, self.epsilon / self.decay)

    def serialize(self):
        return {
            "name": self.name,
            "epsilon": self.epsilon
        }

    def load_vars(self, vars: Dict[str, Any]):
        self.epsilon = vars["epsilon"]
