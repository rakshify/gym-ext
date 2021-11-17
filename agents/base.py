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
        q_val, action = self.get_qval_action(state)
        return action

    def get_qval_action(self, state: Union[int, np.ndarray]) -> Tuple[np.ndarray, int]:
        qvals = self.model.predict(state)
        return self.policy.get_action(qvals)

    def update_model(self, update: Any):
        self.model.update(update)

    def q_grad(self, state: Union[int, np.ndarray], action: int) -> np.ndarray:
        return self.model.grad(state, action)

    def explore_policy(self):
        self.policy.explore()

    @property
    def vec_shape(self):
        return self.model.vec_shape

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
            algorithm.solve_episode(self.env, self, discount_factor)
            self.policy.exploit()
            msg = (f"Finished episode {i} in "
                f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
        # print(self.model.w)
        # print("+" * 80)
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
        if model is not None:
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


class DQNAgent(ModelFreeAgent):
    """This is the base class for all dqn agents."""
    name = ""

    def __init__(self, env: Env, policy: str, model: str = "dqn",
                 verbose: bool = False):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            policy (str): The policy to use.
            model (str): The model to use. Fixed to dqn model
            verbose (bool): Whether to print out information.
        """
        super(DQNAgent, self).__init__(env, policy, model, verbose)
        self.train_after = 1000
        model = self._get_model()
        self.model.init_vars(model=model)

    def train(self, discount_factor: float = 1.0,
              num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            discount_factor (float): The discount factor.
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        X = []
        Y = []
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()
            while True:
                alpha += 1
                if alpha % 10000 == 0:
                    # sepsilon += 1
                    self.explore_policy()
                X.append(state)
                qval, action = self.get_qval_action(state)
                state_, reward, done, _ = self.env.step(action)
                max_next_qval, action_ = agent.get_qval_action(state_)
                y = reward + discount_factor * max_next_qval * np.invert(done)
            self.policy.exploit()
            msg = (f"Finished episode {i} in "
                f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def _get_model(self):
        raise NotImplementedError("Base DQN agent can not make a model.")
