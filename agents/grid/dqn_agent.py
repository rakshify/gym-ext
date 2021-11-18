import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple, Union

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, backend as K

from agents.base import DQNAgent
from algorithms import Algorithm
from gym_ext.envs import Env
from models.dqn import DQN


def mish(x):
    return x*K.tanh(K.softplus(x))


class DQNGridAgent(DQNAgent):
    name = "DQNGrid"

    @property
    def n_features(self):
        return self.env.observation_space.high.shape[0]

    def _get_model(self):
        model = keras.Sequential([
            keras.Input(shape=(self.n_features,)),
            # layers.Dense(128, activation=mish, kernel_initializer='he_normal'),
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.Activation(mish),
            # layers.Dense(64, activation=mish, kernel_initializer='he_normal'),
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.Activation(mish),
            # layers.Dense(64, activation=mish, kernel_initializer='he_normal'),
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.Activation(mish),
            layers.Dense(self.env.action_space.n, kernel_initializer='he_normal')
            # layers.Dense(self.env.action_space.n, activation="softmax", kernel_initializer='he_normal')
        ])
        return model
