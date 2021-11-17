import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from agents.base import DQNAgent
from algorithms import Algorithm
from gym_ext.envs import Env
from models.dqn import DQN


class DQNGridAgent(DQNAgent):
    name = "DQNGrid"

    @property
    def n_features(self):
        return self.env.observation_space.high.shape[0]

    def _get_model(self):
        model = keras.Sequential(
            keras.Input(shape=(self.n_features,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.env.action_space.n, activation="sigmoid")
        )
        return model
