"""Implements the SARSA algorithm."""

import numpy as np

from gym_ext.envs import Env
from algorithms.algorithm import Algorithm


class Sarsa(Algorithm):
    """Implements the SARSA algorithm."""

    name = "sarsa"

    def solve_episode(self, env: Env, agent, df: float) -> float:
        """
        Solve a single episode using the SARSA algorithm.

        Args:
            env: The environment to solve.
            agent: Agent to solve the environment.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        state = env.reset()
        qval, action = agent.get_qval_action(state)
        alpha = 0
        cum_reward = 0
        while True:
            alpha += 1
            if alpha % 10000 == 0:
                agent.explore_policy()
            state_, reward, done, _ = env.step(action)
            next_qval, action_ = agent.get_qval_action(state_)
            update = reward + df * next_qval - qval
            agent.update_model(
                update * agent.q_grad(state, action) * (1 / alpha))
            state, action, qval = state_, action_, next_qval
            cum_reward += reward
            if done:
                break
        return cum_reward


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
        qvals, action = agent.get_qval_action(state)
        qval = qvals[action]
        alpha = 0
        # sepsilon = 0
        cum_reward = 0
        elig_traces = np.zeros(agent.vec_shape)
        while True:
            alpha += 1
            if alpha % 10000 == 0:
                # sepsilon += 1
                agent.explore_policy()
            state_, reward, done, _ = env.step(action)
            next_qvals, action_ = agent.get_qval_action(state_)
            next_qval = next_qvals[action_]
            update = reward + df * next_qval - qval
            grad = agent.q_grad(state, action)
            elig_traces = df * self.lamda * elig_traces + grad
            agent.update_model(update * elig_traces * (1 / alpha))
            state, action, qval = state_, action_, next_qval
            cum_reward += reward
            if done:
                break
        return cum_reward
