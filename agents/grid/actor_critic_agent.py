"""Implements actor-critic agent."""

from agents.base import ActorCriticAgent


class ActorCriticAgent(ActorCriticAgent):
    """Implements actor-critic agent for base grid env."""

    name = "ActorCriticGrid"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.n

    @property
    def n_actions(self):
        """Get the number of actions."""
        return self.env.action_space.n


class ContActorCriticAgent(ActorCriticAgent):
    """Implements actor-critic agent for cont grid env."""

    name = "ContActorCriticGrid"

    @property
    def n_features(self):
        """Get the number of features in state."""
        return self.env.observation_space.high.shape[0]

    @property
    def n_actions(self):
        """Get the number of actions."""
        return self.env.action_space.n
