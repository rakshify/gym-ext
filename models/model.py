"""Implements base model class."""

from typing import Any, Dict, Union

import numpy as np


class Model(object):
    """Implements base model class."""

    name = ''

    def init_vars(self, num_features: int = None, nA: int = None, **kwargs):
        """
        Initialize the model's variables.

        Args:
            num_features: Number of features.
            nA: Number of actions.
            kwargs: Additional arguments.
        """
        raise NotImplementedError("No variables to initialize for base model.")

    def predict(self, state: Any) -> Any:
        """
        Predict the Q-values for a given state.

        Args:
            state: The state.

        Returns:
            The Q-values for the given state.
        """
        raise NotImplementedError("Base model can not predict.")

    def update(self, update: Union[np.ndarray, float], **kwargs):
        """
        Update the model's variables.

        Args:
            update: The update.
            kwargs: Additional arguments.
        """
        raise NotImplementedError("Base model can not update.")

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
        raise NotImplementedError("Base model can not compute gradient.")

    @property
    def vec_shape(self):
        """Model vector shape."""
        raise NotImplementedError("Base model can not compute vector shape.")

    def serialize(self) -> Dict[str, str]:
        """Serialize the model's variables."""
        return {"name": self.name}

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        """
        Save the model's variables.

        Args:
            model_dir: The model directory.

        Returns:
            A dictionary containing the model's variables.
        """
        raise NotImplementedError("Base model can not save variables.")

    def load_vars(self, vars: Dict[str, str]):
        """
        Load the model's variables.

        Args:
            vars: A dictionary containing the model's variables.
        """
        raise NotImplementedError("Base model can not load variables.")
