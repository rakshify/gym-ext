"""All core agents are defined here."""

from agents.base.agent import Agent
from agents.base.actor_critic import ActorCriticAgent
from agents.base.policy_agents import PolicyAgent, ModelFreePolicyAgent
from agents.base.value_agents import (
    ValueAgent, ModelFreeValueAgent, AlgorithmBasedAgent)
from agents.base.nn_agents import DQNAgent


__all__ = [
    "Agent",
    "ActorCriticAgent",
    "PolicyAgent",
    "ModelFreePolicyAgent",
    "ValueAgent",
    "ModelFreeValueAgent",
    "AlgorithmBasedAgent",
    "DQNAgent"
]
