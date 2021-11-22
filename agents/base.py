"""Implements the base class for all agents."""

import os
import random
import time

from typing import Any, Dict, Tuple, Union

import numpy as np

from algorithms import Algorithm
from gym_ext.envs import Env
from models import get_model_by_name
from policies import get_policy_by_name


class Agent(object):
    """Base class for all agents."""

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

    def get_qval_action(self, state: Union[int, np.ndarray]
                        ) -> Tuple[np.ndarray, int]:
        """
        Get the q-value and action for a state.

        Args:
            state (Union[int, np.ndarray]): The state to get the q-value
                                            and action for.

        Returns:
            Tuple[np.ndarray, int]: The q-value and action.
        """
        qvals = self.model.predict(state)
        return qvals, self.policy.get_action(qvals)

    def update_model(self, update: Any):
        """
        Update the model.

        Args:
            update (Any): The update to apply.
        """
        self.model.update(update)

    def q_grad(self, state: Union[int, np.ndarray], action: int) -> np.ndarray:
        """
        Get the gradient of the q-value for a state and action.

        Args:
            state (Union[int, np.ndarray]): The state to get the gradient for.
            action (int): The action to get the gradient for.

        Returns:
            np.ndarray: The gradient of the q-value for the state and action.
        """
        return self.model.grad(state, action)

    def explore_policy(self):
        """Explore the policy."""
        self.policy.explore()

    @property
    def vec_shape(self):
        """Get the shape of the model weights."""
        return self.model.vec_shape

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata["agent"] = {
            "name": self.name,
            "policy": self.policy.serialize()
        }
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env) -> "Agent":
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        policy = metadata["policy"]
        agent = cls(env, policy["name"])
        agent.policy.load_vars(policy)
        return agent


class ValueAgent(Agent):
    """Base class for value agents."""
    
    pass


class PolicyAgent(Agent):
    """Base class for policy agents."""
    
    pass


class ModelFreeAgent(ValueAgent):
    """Base class for all model-free agents."""

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
        episode_rewards = []
        for i in range(num_episodes):
            st = time.time()
            er = algorithm.solve_episode(self.env, self, discount_factor)
            episode_rewards.append(er)
            self.policy.exploit()
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms "
                   f"with reward = {er}.")
            print(msg)
        # print(self.model.w)
        # print("+" * 80)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
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
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        policy = metadata["policy"]
        model = metadata["model"]
        agent = cls(env, policy["name"], model["name"])
        agent.policy.load_vars(policy)
        agent.model.load_vars(model["vars"])
        return agent


class DQNAgent(ModelFreeAgent):
    """Base class for all dqn agents."""

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
        self.target_model = get_model_by_name(model)()
        self.train_after = 1000
        model = self._get_model()
        self.model.init_vars(model=model)
        self.target_model.init_vars(model=model)
        self.batch_size = 32

    def transfer_model_weights(self):
        """Transfer the model weights to the target model."""
        self.target_model.update_weights_from_model(self.model)

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
        batch = []
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()
            while True:
                qval, action = self.get_qval_action(state)
                state_, reward, done, _ = self.env.step(action)
                r = reward if not done else -100
                # Get next best qval from target model
                next_qvals = self.target_model.predict(state_)
                max_next_qval = np.max(next_qvals)
                y = r + discount_factor * max_next_qval * np.invert(done)
                qval[action] = y
                batch.append((state, qval))
                if len(batch) >= self.train_after:
                    b = random.sample(batch, min(len(batch), self.batch_size))
                    X, Y = zip(*b)
                    self.model.train(np.array(X), np.array(Y))
                if i % 100 == 0:
                    self.env.render()
                self.policy.exploit()
                if done:
                    self.transfer_model_weights()
                    break
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
            print(f"Data size till now = {len(batch)}")
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def _get_model(self):
        """Get the neural network to use for training."""
        raise NotImplementedError("Base DQN agent can not make a model.")


class ModelFreePolicyAgent(PolicyAgent):
    """Base class for all model-free policy agents."""

    name = ""

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        return self.policy.get_action(state)

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(ModelFreePolicyAgent, self).update_metadata(metadata)
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        model_dir = os.path.join(model_dir, "agent-policy-vars")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        meta = {"vars": self.policy.save_vars(model_dir)}
        metadata["agent"]["policy"].update(meta)
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env
                       ) -> "ModelFreePolicyAgent":
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        policy = metadata["policy"]
        agent = cls(env, policy["name"])
        agent.policy.load_vars(policy["vars"])
        return agent

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
        self.policy.init_vars(self.n_features, self.n_actions)
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()

            # Step 1: Collect history
            history = []
            while True:
                action = self.get_action(state)
                state_, reward, done, _ = self.env.step(action)
                # r = reward if not done else -100
                update = self.policy.grad(state, action)
                history.append((state, action, reward, update))
                state = state_
                if done:
                    break

            # Step 2: Cumulative rewards for state values
            cum_rewards = []
            cum_rewards.append(history[-1][2])
            states, actions, rewards, updates = zip(*history)
            for state, action, reward, update in reversed(history):
                cum_rewards.append(reward + cum_rewards[-1])
            cum_rewards = np.array(cum_rewards)[::-1]

            # Step 3: Update policy
            alpha = 0.0
            print(f"Steps in this episode = {len(history)}")
            history = zip(states, actions, cum_rewards, updates)
            for state, action, reward, update in history:
                alpha += 1
                self.policy.update_policy((1 / alpha) * update * reward)
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")
