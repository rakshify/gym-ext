"""Implements the base class for all agents."""

import os
import random
import time

from typing import Any, Dict

import numpy as np

from gym_ext.envs import Env


class Agent(object):
    """Base class for all agents."""

    name = ""

    def __init__(self, env: Env, verbose: bool = False, **kwargs):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            verbose (bool): Whether to print out information.
            **kwargs: Additional arguments.
        """
        self.env = env
        self.verbose = verbose

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        raise NotImplementedError("Base agent can not decide on action.")

    def train(self, num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError("Base agent can not train.")

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata["agent"] = {"name": self.name}
        return metadata
