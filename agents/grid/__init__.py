"""All agents for grid env."""

from agents.grid.agent import GridAgent
from agents.grid.actor_critic_agent import ActorCriticAgent, ContActorCriticAgent
from agents.grid.continuous_agent import ContGridAgent
from agents.grid.dqn_agent import DQNGridAgent
from agents.grid.sgd_policy_agent import SGDPolicyAgent, ContSGDPolicyAgent


_ALL_AGENTS = [
    GridAgent, ContGridAgent, DQNGridAgent, SGDPolicyAgent, ContSGDPolicyAgent,
    ActorCriticAgent, ContActorCriticAgent
]
