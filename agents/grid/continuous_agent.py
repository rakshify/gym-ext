"""Implements an agent for continuous grid env."""

from agents.grid.agent import GridAgent


class ContGridAgent(GridAgent):
    """Implements an agent for continuous grid env."""

    name = "ContGridWorld"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.high.shape[0]
