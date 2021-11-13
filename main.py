"""Example runs to run the environment using solvers."""

import argparse
import os
import sys

# import gym

from gym_ext import gym


def main():
    """Main function."""
    env = gym.make('GridWorld-v0')
    state = env.reset()
    steps = 0
    preState = None
    ACTIONS = [" ^ ", " v ", " < ", " > "]
    while True:
        steps += 1
        action = env.action_space.sample()
        preState = state
        state, reward, done, info = env.step(action)
        text = ""
        for i in range(env.states.shape[0]):
            if i % 4 == 0:
                text += "\n"
            if env.states[i] == "t":
                text += " * "
            elif env.states[i] == "x" or i == state:
                text += f" {env.states[i]} "
            elif i == preState:
                text += ACTIONS[action]
            else:
                text += " - "
        print(text)
        if done or steps > 10:
            break


if __name__ == '__main__':
    main()
