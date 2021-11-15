"""This implements the base class for all algorithms."""

from gym_ext.envs import Env


class Algorithm(object):
    """This is the base class for all algorithms."""
    
    name = ""

    def solve_episode(self, env: Env, agent: "Agent", episode: int,
                      df: float) -> float:
        """
        Solve a single episode using the algorithm.

        Args:
            env: The environment to solve.
            agent: Agent to solve the environment.
            episode: The episode number.
            df: The discount factor to use.

        Returns:
            The total reward for the episode.
        """
        raise NotImplementedError("Base algorithm not callable")
