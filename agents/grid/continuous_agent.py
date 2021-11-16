import argparse
import json
import os
import sys
import time

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from agents.grid.agent import GridAgent
from algorithms import Algorithm
from gym_ext.envs import Env


class ContGridAgent(GridAgent):
    name = "ContGridWorld"

    @property
    def n_features(self):
        return self.env.observation_space.high.shape[0]
