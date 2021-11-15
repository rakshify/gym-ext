from typing import Any, Dict

from agents.base import Agent
from agents.grid import GridAgent


_ALL_AGENTS = [GridAgent]
_REGISTERED_AGENTS = {a.name: a for a in _ALL_AGENTS}


def get_agent_by_name(name: str) -> Agent:
    if name not in _REGISTERED_AGENTS:
        raise ValueError(f"Agent {name} not found.")
    return _REGISTERED_AGENTS[name]


def load_agent(metadata: Dict[str, Any]):
    pass
