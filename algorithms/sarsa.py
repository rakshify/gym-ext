"""Implements the SARSA algorithm."""

from typing import Any, Callable, Tuple

import numpy as np

from gym_ext.envs import Env
from algorithms.algorithm import Algorithm


class Sarsa(Algorithm):
    """Implements the SARSA algorithm."""

    name = "sarsa"

    def solve_episode(self, env: Env, agent, df: float) -> float:
        """
        Solve a single episode using the SARSA lambda algorithm.

        Args:
            env: The environment to solve.
            agent: Solver to solve the environment.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        state = env.reset()
        qvals = agent.get_qvals(state)
        action = agent.get_action(state)
        qval = qvals[action]
        alpha = 0
        cum_reward = 0
        self.reset_vars()
        while True:
            alpha += 1
            if alpha % 10000 == 0:
                agent.explore_policy()
            state, action, qval, update, reward, done = self._solve_step(
                env.step, agent.get_qvals, agent.get_action, agent.q_grad,
                agent.vec_shape, state, action, qval, df)
            agent.update_model(update * (1 / alpha))
            cum_reward += reward
            if done:
                break
        return cum_reward

    def get_update(self, reward: float, qval: float, nqval: float,
                   grad: np.ndarray, vec_shape: tuple, df: float) -> float:
        """
        Get the update for the agent model.

        Args:
            reward: The reward for the action.
            qval: The current q-value.
            nqval: The next q-value.
            grad: The gradient of the agent model.
            vec_shape: The shape of the agent model weight vector.
            df: The discount factor to use.

        Returns:
            The update for the agent model.
        """
        update = reward + df * nqval - qval
        return update * grad

    def reset_vars(self):
        pass

    def _solve_step(self, step_func: Callable, qval_func: Callable,
                   act_func: Callable, grad_func: Callable, vec_shape: tuple,
                   state: Any, action: Any, qval: float, df: float
                   ) -> Tuple[Any, Any, float, np.ndarray, float, bool]:
        """
        Solve a single step of the algorithm.

        Args:
            step_func: Function to call to step the environment.
            qval_func: Function to call to get the qvals for the state.
            act_func: Function to call to get the action for the state.
            grad_func: Function to call to get the agent model gradient.
            vec_shape: Shape of the agent model weight vector.
            state: The state to solve.
            action: The action to solve.
            qval: The current q-value.
            df: The discount factor to use.

        Returns:
            The new state, action, q-value, the update, the reward and done.
        """
        state_, reward, done, _ = step_func(action)
        nqvals = qval_func(state_)
        action_ = act_func(state_)
        nqval = nqvals[action_]
        grad = grad_func(state, action)
        update = self.get_update(reward, qval, nqval, grad, vec_shape, df)
        print(update)
        return state_, action_, nqval, update, reward, done

    def serialize(self) -> dict:
        """Serialize the algorithm."""
        return {"name": self.name}

    def load_vars(self, vars: dict):
        pass


class SarsaLambda(Sarsa):
    """Implements the SARSA-lambda algorithm."""

    name = "sarsa_lambda"

    def __init__(self, lamda: float = 0.7):
        """
        Initialize the SARSA lambda agent.

        Args:
            lamda: The lambda value to use.
        """
        self.lamda = lamda
        self.elig_traces = None

    def get_update(self, reward: float, qval: float, nqval: float,
                   grad: np.ndarray, vec_shape: tuple, df: float) -> float:
        """
        Get the update for the agent model.

        Args:
            reward: The reward for the action.
            qval: The current q-value.
            nqval: The next q-value.
            grad: The gradient of the agent model.
            vec_shape: The shape of the agent model weight vector.
            df: The discount factor to use.

        Returns:
            The update for the agent model.
        """
        update = reward + df * nqval - qval
        if self.elig_traces is None:
            self.elig_traces = np.zeros(vec_shape)
        self.elig_traces = df * self.lamda * self.elig_traces + grad
        return update * self.elig_traces

    def reset_vars(self):
        self.elig_traces = None

    # def solve_episode(self, env: Env, agent, df: float) -> float:
    #     """
    #     Solve a single episode using the SARSA lambda algorithm.

    #     Args:
    #         env: The environment to solve.
    #         agent: Solver to solve the environment.
    #         df: The discount factor to use.

    #     Returns:
    #         The total reward for the episode.
    #     """
    #     state = env.reset()
    #     qvals = agent.get_qvals(state)
    #     action = agent.get_action(state)
    #     qval = qvals[action]
    #     alpha = 0
    #     # sepsilon = 0
    #     cum_reward = 0
    #     while True:
    #         alpha += 1
    #         if alpha % 10000 == 0:
    #             # sepsilon += 1
    #             agent.explore_policy()
    #         state, action, qval, update, reward, done = self.solve_step(
    #             env.step, agent.get_qvals, agent.get_action, agent.q_grad
    #             agent.vec_shape, state, action, qval, df)
    #         agent.update_model(update * (1 / alpha))

    #         # state_, reward, done, _ = env.step(action)
    #         # next_qvals, action_ = agent.get_qval_action(state_)
    #         # next_qval = next_qvals[action_]
    #         # update = reward + df * next_qval - qval
    #         # grad = agent.q_grad(state, action)
    #         # elig_traces = df * self.lamda * elig_traces + grad
    #         # agent.update_model(update * elig_traces * (1 / alpha))
    #         # state, action, qval = state_, action_, next_qval

    #         cum_reward += reward
    #         if done:
    #             break
    #     return cum_reward

    def serialize(self) -> dict:
        """Serialize the algorithm."""
        return {"name": self.name, "lamda": self.lamda}

    def load_vars(self, vars: dict):
        """Load the variables of the algorithm."""
        self.lamda = vars["lamda"]
