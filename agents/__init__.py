from typing import Any, Dict

from agents.base import Agent
from agents.grid import GridAgent, ContGridAgent
from gym_ext.envs import Env


_ALL_AGENTS = [GridAgent, ContGridAgent]
_REGISTERED_AGENTS = {a.name: a for a in _ALL_AGENTS}


def get_agent_by_name(name: str) -> Agent:
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    return _REGISTERED_AGENTS[name]


def load_agent(metadata: Dict[str, Any], env: Env):
    meta = metadata["agent"]
    return get_agent_by_name(meta["name"]).load_from_meta(meta, env)
