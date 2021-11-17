"""This implements a deep q-network model with a single weight vector."""

import os

from typing import Dict

import numpy as np

from models.model import Model


class DQN(Model):
    """This implements a deep q-network model with a single weight vector."""
    name = 'dqn'

    def init_vars(self, num_features: int=None, nA: int=None, **kwargs):
        """
        Initialize the model.

        Args:
            num_features: Number of features.
            nA: Number of actions.
            kwargs: Additional arguments.
        """
        self.nn = kwargs["model"]
        self.nn.compile(loss='mse', optimizer=keras.optimizers.Adam())

    def train(self, x: np.ndarray, y: np.ndarray):
        self.nn.train(x, y)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the Q-values for the given state.

        Args:
            state: State.

        Returns:
            Q-values for the given state.
        """
        return self.nn.predict(state)

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        """
        Save the model's variables.
            
        Args:
            model_dir: The model directory.

        Returns:
            A dictionary containing the model's variables.
        """
        model_path = model_dir if model_dir.endswith("/") else model_dir + "/"
        vars = {"model_weights": model_path}
        self.nn.save_weights(vars["model_weights"])
        return vars

    def load_vars(self, vars: Dict[str, str]):
        """
        Load the model's variables.

        Args:
            vars: A dictionary containing the model's variables.
        """
        self.nn.load_weights(vars["model_weights"])
        self.nn.compile(loss='mse', optimizer=keras.optimizers.Adam())