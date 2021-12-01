"""Implements softmax policy."""

import os

from typing import Dict

import numpy as np

from policies.policy import Policy


class Softmax(Policy):
    """Implements softmax policy."""

    name = 'softmax'

    def init_vars(self, num_features: int, nA: int, **kwargs):
        """
        Initialize the parameters.

        Args:
            num_features: Number of features.
            nA: Number of actions.
            kwargs: Additional arguments.
        """
        self.w = np.random.random((nA, num_features))

    def get_action(self, state: np.ndarray) -> int:
        """
        Get the action to take.

        Args:
            state: The current state.

        Returns:
            The action to take.
        """
        pi = self._action_probabilities(state)
        return np.random.choice(len(pi), p=pi)

    def update_policy(self, update: np.ndarray, **kwargs):
        """
        Update the policy.

        Args:
            update: The update to apply.
        """
        self.w += update

    def grad(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the gradient of the score function wrt the policy's parameters.

        Args:
            state: The state.
            action: The action.

        Returns:
            The gradient of the score function wrt the policy's parameters.
        """
        pi = self._action_probabilities(state)
        gradient = np.zeros(self.w.shape)
        gradient[action, :] = state
        for i in range(len(pi)):
            gradient[i, :] -= state * pi[i]
        return gradient

    def _action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get the action probabilities.

        Args:
            state: The current state.

        Returns:
            The action probabilities.
        """
        vals = np.dot(self.w, state)
        vmax = np.max(vals)
        exp = np.exp(vals - vmax)
        return exp / np.sum(exp)

    # @property
    # def vec_shape(self) -> Tuple[int]:
    #     """Get the shape of the parameter vector."""
    #     return self.w.shape

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        """
        Save the policy's parameters.

        Args:
            model_dir: The model directory.

        Returns:
            A dictionary containing the policy's parameters.
        """
        vars = {"w": os.path.join(model_dir, "w.npy")}
        np.save(vars["w"], self.w)
        return vars

    def load_vars(self, vars: Dict[str, str]):
        """
        Load the policy's parameters.

        Args:
            vars: A dictionary containing the policy's parameters.
        """
        self.w = np.load(vars["w"])
