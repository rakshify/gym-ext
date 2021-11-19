"""Implements base grid agent."""

from agents.base import ModelFreeAgent


class GridAgent(ModelFreeAgent):
    """Implements base grid agent."""

    name = "GridWorld"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.n
