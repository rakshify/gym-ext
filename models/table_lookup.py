"""Implements a lookup table kind of the model."""

import os

from typing import Dict

import numpy as np

from models.model import Model


class TableLookup(Model):
    """Implements a lookup table kind of the model."""

    name = 'table_lookup'

    def init_vars(self, num_features: int, nA: int, **kwargs):
        """
        Initialize the model's variables.

        Args:
            num_features: Number of features.
            nA: Number of actions.
            kwargs: Additional arguments.
        """
        self.q_vals = np.zeros((num_features, nA))

    def predict(self, state: int) -> np.ndarray:
        """
        Predict the Q-values for a given state.

        Args:
            state: The state.

        Returns:
            The Q-values for the given state.
        """
        return self.q_vals[state, :]

    def update(self, update: np.ndarray, **kwargs):
        """
        Update the model's variables.

        Args:
            update: The update.
            kwargs: Additional arguments.
        """
        self.q_vals += update

    def grad(self, state: int, action: int) -> np.ndarray:
        """
        Compute the gradient of the model's output wrt the model's variables.

        Args:
            state: The state.
            action: The action.

        Returns:
            The gradient of the model's output with respect to the model's
            variables.
        """
        gradient = np.zeros(self.q_vals.shape)
        gradient[state, action] = 1
        return gradient

    @property
    def vec_shape(self):
        """Model vector shape."""
        return self.q_vals.shape

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        """
        Save the model's variables.

        Args:
            model_dir: The model directory.

        Returns:
            A dictionary containing the model's variables.
        """
        vars = {"qvals": os.path.join(model_dir, "qvals.npy")}
        np.save(vars["qvals"], self.q_vals)
        return vars

    def load_vars(self, vars: Dict[str, str]):
        """
        Load the model's variables.

        Args:
            vars: A dictionary containing the model's variables.
        """
        self.q_vals = np.load(vars["qvals"])
