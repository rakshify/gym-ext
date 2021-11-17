"""This implements a linear model with a single weight vector."""

import os

from typing import Dict

import numpy as np

from models.model import Model


class Linear(Model):
    """This implements a linear model with a single weight vector."""
    name = 'linear'

    def init_vars(self, num_features: int=None, nA: int=None, **kwargs):
        """
        Initialize the weight vector.

        Args:
            num_features: Number of features.
            nA: Number of actions.
            kwargs: Additional arguments.
        """
        self.w = np.zeros((nA, num_features))

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the Q-values for the given state.

        Args:
            state: State.

        Returns:
            Q-values for the given state.
        """
        return np.dot(self.w, state)

    def update(self, update: np.ndarray, **kwargs):
        """
        Update the model's variables.

        Args:
            update: The update.
            kwargs: Additional arguments.
        """
        self.w += update
        # print(self.w)
        # print("+" * 80)

    def grad(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the gradient of the model's output with respect to the
        model's variables.

        Args:
            state: The state.
            action: The action.

        Returns:
            The gradient of the model's output with respect to the model's
            variables.
        """
        gradient = np.zeros(self.w.shape)
        gradient[action, :] = state
        return gradient

    @property
    def vec_shape(self):
        """Model vector shape."""
        return self.w.shape

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        """
        Save the model's variables.
            
        Args:
            model_dir: The model directory.

        Returns:
            A dictionary containing the model's variables.
        """
        vars = {"w": os.path.join(model_dir, "w.npy")}
        np.save(vars["w"], self.w)
        return vars

    def load_vars(self, vars: Dict[str, str]):
        """
        Load the model's variables.

        Args:
            vars: A dictionary containing the model's variables.
        """
        self.w = np.load(vars["w"])