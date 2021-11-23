"""All policies."""

from policies.policy import Policy
from policies.softmax import Softmax

_ALL_POLICIES = [Softmax]
_REGISTERED_POLICIES = {p.name: p for p in _ALL_POLICIES}


def get_policy_by_name(name: str) -> Policy:
    """
    Get policy by name.

    Args:
        name: Name of the policy.

    Returns:
        Policy object.
    """
    if name not in _REGISTERED_POLICIES:
        raise ValueError(f"Policy {name} not found")
    return _REGISTERED_POLICIES[name]
