"""Implements base policy."""

from typing import List


class Policy(object):
    """Implements base policy."""

    name = ''

    def get_action(self, values: List[float]) -> int:
        """
        Get the action to take.

        Args:
            values: The values of all the actions.

        Returns:
            The action to take.
        """
        raise NotImplementedError("Base policy can not decide on action.")

    def exploit(self, **kwargs):
        """Reduce epsilon to exploit the best action."""
        raise NotImplementedError("Base policy can not change vars.")

    def explore(self, **kwargs):
        """Increase epsilon to explore the environment."""
        raise NotImplementedError("Base policy can not change vars.")

    def serialize(self):
        """Serialize the variables."""
        return {"name": self.name}
