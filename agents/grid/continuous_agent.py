"""Implements an agent for continuous grid env."""

from agents.base import AlgorithmBasedAgent


class ContGridAgent(AlgorithmBasedAgent):
    """Implements an agent for continuous grid env."""

    name = "ContGridWorld"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.high.shape[0]

    @property
    def n_actions(self):
        """Get the number of actions."""
        return self.env.action_space.n
