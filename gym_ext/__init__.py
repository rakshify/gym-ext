"""Gym environment extension."""

import json
import os

import gym

from gym_ext.envs import _ALL_ENVS

for env in _ALL_ENVS:
    gym.envs.registration.register(
        id=f'{env.name}-{env.version}',
        entry_point=env.entry_point,
    )

base_dir = os.path.abspath(os.path.dirname(__file__))
pkg_file = os.path.join(base_dir, "package_info.json")
with open(pkg_file) as f:
    pkg_info = json.load(f)

__version__ = pkg_info["version"]
__author__ = pkg_info["author"]
__credits__ = pkg_info["credits"]
