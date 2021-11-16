"""This implements the base class for all agents."""

import os
import time

from typing import Any, Dict

from algorithms import Algorithm
from gym_ext.envs import Env
from models import get_model_by_name
from policies import get_policy_by_name


class Agent(object):
    """This is the base class for all agents."""
    name = ""

    def __init__(self, env: Env, policy: str, verbose: bool = False):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            policy (str): The policy to use.
            verbose (bool): Whether to print out information.
        """
        self.env = env
        self.policy_name = policy
        self.policy = get_policy_by_name(policy)()
        self.verbose = verbose

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        raise NotImplementedError("Base agent can not decide action to take")

    def train(self, algorithm: Algorithm, discount_factor: float = 1.0,
              num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            algorithm (Algorithm): The algorithm to use.
            discount_factor (float): The discount factor.
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        self.model.init_vars(self.n_features, self.env.action_space.n)
        for i in range(num_episodes):
            st = time.time()
            self.policy.exploit()
            algorithm.solve_episode(self.env, self, discount_factor)
            msg = (f"Finished episode {i} in "
                f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def update_metadata(self, metadata: Dict[str, Any]):
        metadata["agent"] = {
            "name": self.name,
            "policy": self.policy.serialize()
        }
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env) -> "Agent":
        policy = metadata["policy"]
        agent = cls(env, policy["name"])
        agent.policy.load_vars(policy)
        return agent


class ModelFreeAgent(Agent):
    """This is the base class for all model-free agents."""
    name = ""

    def __init__(self, env: Env, policy: str, model: str,
                 verbose: bool = False):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            policy (str): The policy to use.
            model (str): The model to use.
            verbose (bool): Whether to print out information.
        """
        super(ModelFreeAgent, self).__init__(env, policy, verbose)
        self.model_name = model
        self.model = get_model_by_name(model)()

    def update_metadata(self, metadata: Dict[str, Any]):
        metadata = super(ModelFreeAgent, self).update_metadata(metadata)
        meta = self.model.serialize()
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        model_dir = os.path.join(model_dir, "agent-model-vars")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        meta["vars"] = self.model.save_vars(model_dir)
        metadata["agent"]["model"] = meta
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env
                       ) -> "ModelFreeAgent":
        policy = metadata["policy"]
        model = metadata["model"]
        agent = cls(env, policy["name"], model["name"])
        agent.policy.load_vars(policy)
        agent.model.load_vars(model["vars"])
        return agent
