"""Implements stochastic policy gradient agent."""

from agents.base import ModelFreePolicyAgent


class SGDPolicyAgent(ModelFreePolicyAgent):
    """Implements stochastic policy gradient agent."""

    name = "SGDPolicyGrid"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.high.shape[0]

    @property
    def n_actions(self):
        """Get the number of actions."""
        return self.env.action_space.n
