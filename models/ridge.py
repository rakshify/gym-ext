"""Implements a ridge model with a single weight vector."""

import numpy as np

from models.linear import Linear


class Ridge(Linear):
    """Implements a ridge model with a single weight vector."""

    name = 'ridge'

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the Q-values for the given state.

        Args:
            state: State.

        Returns:
            Q-values for the given state.
        """
        return np.dot(self.w, state)

    def grad(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the gradient of the model's output wrt the model's variables.

        Args:
            state: The state.
            action: The action.

        Returns:
            The gradient of the model's output with respect to the model's
            variables.
        """
        gradient = np.zeros(self.w.shape)
        gradient[action, :] = state - 0.5 * self.w[action, :]
        # gradient = -0.5 * self.w
        # gradient[action, :] += state
        return gradient
