"""Implements greedy policy."""

from typing import Any, Dict, List

import numpy as np

from policies.policy import Policy


class Greedy(Policy):
    """Implements greedy policy."""

    name = 'greedy'

    def __init__(self, epsilon: float = 0.99, decay: float = 0.99,
                 min_eps: float = 0.01):
        """
        Initialize the greedy policy.

        Args:
            epsilon: The probability of choosing a random action.
            decay: The decay rate of epsilon.
            min_eps: The minimum value of epsilon.
        """
        self.epsilon = epsilon
        self.decay = decay
        self.min_eps = min_eps

    def get_action(self, values: List[float]) -> int:
        """
        Get the action to take.

        Args:
            values: The values of all the actions.

        Returns:
            The action to take.
        """
        if np.random.rand() > self.epsilon:
            return np.argmax(values)
        else:
            return np.random.choice(len(values))

    def exploit(self, **kwargs):
        """Reduce epsilon to exploit the best action."""
        self.epsilon *= self.decay
        self.epsilon = max(self.min_eps, self.epsilon)

    def explore(self, **kwargs):
        """Increase epsilon to explore the environment."""
        self.epsilon = min(0.99, self.epsilon / self.decay)

    def serialize(self):
        """Serialize the variables."""
        return {
            "name": self.name,
            "epsilon": self.epsilon
        }

    def load_vars(self, vars: Dict[str, Any]):
        """Load the variables from the checkpoint."""
        self.epsilon = vars["epsilon"]
