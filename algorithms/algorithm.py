"""Implements the base class for all algorithms."""

from gym_ext.envs import Env


class Algorithm(object):
    """Base class for all algorithms."""

    name = ""

    def solve_episode(self, env: Env, agent, df: float) -> float:
        """
        Solve a single episode using the algorithm.

        Args:
            env: The environment to solve.
            agent: Agent to solve the environment.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        raise NotImplementedError("Base algorithm not callable")
