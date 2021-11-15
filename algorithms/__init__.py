from algorithms.algorithm import Algorithm
from algorithms.sarsa import Sarsa, SarsaLambda


_ALL_ALGORITHMS = [Sarsa, SarsaLambda]
_REGISTERED_ALGORITHMS = {a.name: a for a in _ALL_ALGORITHMS}


def get_algorithm_by_name(name: str) -> Algorithm:
    if name not in _REGISTERED_ALGORITHMS:
        raise ValueError(f"Algorithm {name} not found.")
    return _REGISTERED_ALGORITHMS[name]
