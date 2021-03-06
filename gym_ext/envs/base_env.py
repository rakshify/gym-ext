"""Implements the base environment."""

from typing import Any, Dict, Tuple

import gym


class Env(gym.Env):
    """Implements the base environment."""

    name = ""
    version = None
    entry_point = None

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Play an action in a state.

        Args:
            action: Action to take.

        Returns:
            A tuple of (next state, reward, status, extra info).
        """
        err = "Base env can not take a step based on the action."
        raise NotImplementedError(err)

    def reset(self) -> int:
        """Start a new episode by sampling a new state."""
        raise NotImplementedError("Base env can not sample a new state")

    def update_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the metadata.

        Args:
            metadata: A dictionary of metadata.

        Returns:
            A dictionary of updated metadata.
        """
        metadata["env"] = {"name": self.name}
        return metadata

    @classmethod
    def load_from_meta(cls, meta: Dict[str, Any]) -> "Env":
        """
        Load Env from metadata.

        Args:
            meta: A dictionary of metadata.

        Returns:
            An Env.
        """
        return gym.make(meta["env"]["name"])
