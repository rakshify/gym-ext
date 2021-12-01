"""All agents."""

from typing import Any, Dict

from agents.base import Agent, ActorCriticAgent
from agents.grid import _ALL_AGENTS as _ALL_GRID_AGENTS
from gym_ext.envs import Env


_ALL_AGENTS = _ALL_GRID_AGENTS
_REGISTERED_AGENTS = {a.name: a for a in _ALL_AGENTS}


def get_agent_by_name(name: str, env, **kwargs) -> Agent:
    """Get an agent by name."""
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    agent = _REGISTERED_AGENTS[name](env, **kwargs)
    if isinstance(agent, ActorCriticAgent):
        actor = None
        akwargs = kwargs.get("actor", {})
        if "name" in akwargs:
            actor = get_agent_by_name(env=env, **akwargs)
        critic = None
        ckwargs = kwargs.get("critic", {})
        if "name" in ckwargs:
            critic = get_agent_by_name(env=env, **ckwargs)
        agent.set_actor_critic(actor, critic)
    return agent


def load_agent(metadata: Dict[str, Any], env: Env):
    """Load an agent from metadata."""
    meta = metadata["agent"]
    name = meta["name"]
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    agent = _REGISTERED_AGENTS[name].load_from_meta(meta, env)
    if isinstance(agent, ActorCriticAgent):
        actor = load_agent(meta["actor_meta"], env)
        critic = load_agent(meta["critic_meta"], env)
        agent.set_actor_critic(actor, critic)
    return agent
