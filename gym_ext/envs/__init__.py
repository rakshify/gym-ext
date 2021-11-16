"""All environments for the gym_ext package."""

from typing import Any, Dict

from gym_ext.envs.base_env import Env
from gym_ext.envs.grid import GridEnv, OneHotGridEnv

_ALL_ENVS = [GridEnv, OneHotGridEnv]
_REGISTERED_ENVS = {e.name: e for e in _ALL_ENVS}


def load_env(metadata: Dict[str, Any]) -> Env:
    """
    Loads an environment from a model directory.

    Args:
        metadata: A dictionary of metadata.

    Returns:
        The environment.
    """
    env_name = metadata["env"]["name"]
    if env_name not in _REGISTERED_ENVS:
        raise ValueError(f"Environment {env_name} not found.")
    return _REGISTERED_ENVS[env_name].load_from_meta(metadata["env"])
