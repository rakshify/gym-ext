import time

import numpy as np

# from matplotlib import pyplot as plt

from gym_ext.envs import Env
from algorithms.algorithm import Algorithm


class Sarsa(Algorithm):
    name = "sarsa"

    def solve_episode(self, env: Env, agent: "Agent", episode: int,
                      df: float) -> float:
        """
        Solve a single episode using the SARSA algorithm.

        Args:
            env: The environment to solve.
            agent: Agent to solve the environment.
            episode: The episode number.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        start = time.time()
        eps = 1 / (episode + 1)
        # TODO: Update epsilon for this episode
        # self.policy.update_policy(self.qvals, eps)
        state = env.reset()
        qval, action = agent.get_qval_action(state)
        alpha = 0
        # sepsilon = 0
        cum_reward = 0
        while True:
            # if alpha % 10000 == 0:
            #     sepsilon += 1
            alpha += 1
            # eps = min(1, sepsilon / (episode + 1))
            state_, reward, done, _ = env.step(action)
            next_qval, action_ = agent.get_qval_action(state_)
            update = reward + df * next_qval - qval
            agent.update_model(update * agent.q_grad, (1 / alpha))
            state, action, qval = state_, action_, next_qval
            cum_reward += reward
            # self.policy.update_policy(self.qvals, eps)
            if done:
                break
        msg = (f"Finished episode {episode} in "
               f"{int((time.time() - start) * 100000) / 100}ms.")
        print(msg)
        return cum_reward


class SarsaLambda(Sarsa):
    name = "sarsa_lambda"

    def __init__(self, lamda: float = 0.7):
        """
        Initialize the SARSA lambda agent.

        Args:
            lamda: The lambda value to use.
        """
        self.lamda = lamda

    def solve_episode(self, env: Env, agent: "Agent", episode: int,
                      df: float) -> float:
        """
        Solve a single episode using the SARSA lambda algorithm.

        Args:
            env: The environment to solve.
            agent: Solver to solve the environment.
            episode: The episode number.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        start = time.time()
        eps = 1 / (episode + 1)
        # self.policy.update_policy(self.qvals, eps)
        state = env.reset()
        qval, action = agent.get_qval_action(state)
        alpha = 0
        # sepsilon = 0
        cum_reward = 0
        elig_traces = np.zeros(agent.vec_shape)
        while True:
            # if alpha % 10000 == 0:
            #     sepsilon += 1
            alpha += 1
            # eps = min(1, sepsilon / epsilon)
            state_, reward, done, _ = env.step(action)
            next_qval, action_ = agent.get_qval_action(state_)
            update = reward + df * next_qval - qval
            elig_traces = df * self.lamda + agent.q_grad
            agent.update_model(update * elig_traces, (1 / alpha))
            state, action, qval = state_, action_, next_qval
            cum_reward += reward
            # self.policy.update_policy(self.qvals, eps)
            if done:
                break
        # CODE HERE
        msg = (f"Finished episode {episode} in "
               f"{int((time.time() - start) * 100000) / 100}ms.")
        print(msg)
        return cum_reward
