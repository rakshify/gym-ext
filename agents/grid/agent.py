"""Implements base grid env agent."""

from agents.base import AlgorithmBasedAgent


class GridAgent(AlgorithmBasedAgent):
    """Implements base grid env agent."""

    name = "GridWorld"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.n

    @property
    def n_actions(self):
        """Get the number of actions."""
        return self.env.action_space.n
