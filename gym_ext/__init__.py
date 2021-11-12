from gym.envs.registration import register

from gym_ext.envs import GridEnv

_ALL_ENVS = [GridEnv]

for env in _ALL_ENVS:
    register(
        id=f'{env.name}-{env.version}',
        entry_point=env.entry_point,
    )