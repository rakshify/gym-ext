"""Implements the DQN agent."""

from tensorflow import keras
from keras import layers, backend as K

from agents.base import DQNAgent


def mish(x):
    """Mish activation function."""
    return x*K.tanh(K.softplus(x))


class DQNGridAgent(DQNAgent):
    """Implements the DQN agent."""

    name = "DQNGrid"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.high.shape[0]

    def _get_model(self) -> keras.Model:
        """Get the neural network to use for training."""
        model = keras.Sequential([
            keras.Input(shape=(self.n_features,)),
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.Activation(mish),
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.Activation(mish),
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.Activation(mish),
            layers.Dense(self.env.action_space.n,
                         kernel_initializer='he_normal')
        ])
        return model
