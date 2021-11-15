from policies.policy import Policy
from policies.greedy import Greedy

_ALL_POLICIES = [Greedy]
_REGISTERED_POLICIES = {p.name: p for p in _ALL_POLICIES}


def get_policy_by_name(name: str):
    if name not in _REGISTERED_POLICIES:
        raise ValueError(f"Policy {name} not found")
    return _REGISTERED_POLICIES[name]