"""All agents."""

from typing import Any, Dict

from agents.base import Agent
from agents.grid import _ALL_AGENTS as _ALL_GRID_AGENTS
from gym_ext.envs import Env


_ALL_AGENTS = _ALL_GRID_AGENTS
_REGISTERED_AGENTS = {a.name: a for a in _ALL_AGENTS}


def get_agent_by_name(name: str, env, **kwargs) -> Agent:
    """Get an agent by name."""
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    return _REGISTERED_AGENTS[name](env, **kwargs)


def load_agent(metadata: Dict[str, Any], env: Env):
    """Load an agent from metadata."""
    meta = metadata["agent"]
    name = meta["name"]
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    return _REGISTERED_AGENTS[name].load_from_meta(meta, env)
