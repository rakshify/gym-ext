"""This implements base policy."""

from typing import List, Tuple


class Policy(object):
    """This implements base policy."""
    name = ''

    def get_action(self, values: List[float]) -> Tuple[float, int]:
        raise NotImplementedError("Base policy can not decide on action.")

    def update_policy(self, update: float, **kwargs):
        raise NotImplementedError("Base policy can not update.")
