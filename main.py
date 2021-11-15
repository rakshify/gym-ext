"""Example runs to run the environment using solvers."""

import argparse
import json
import os

# import gym

from agents import get_agent_by_name, load_agent
from algorithms import get_algorithm_by_name
from gym_ext import gym, load_env


def create_argument_parser():
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('--example', '-e',
                        required=True,
                        help="Example to run by name.")
    parser.add_argument('--step', '-s',
                        required=True,
                        help="Train/Predict.")
    parser.add_argument('--model_dir', '-d',
                        required=True,
                        help="Model directory to save or load the model from.")
    return parser


def solve_env(env_name: str, step: str, model_dir: str):
    """Solve the environment."""
    if step == "train":
        env = gym.make(env_name)
        agent = get_agent_by_name(env_name.split("-v")[0])(
            policy="greedy", model="table_lookup")
        algorithm = get_algorithm_by_name("sarsa")()
        agent.train(env, algorithm, num_episodes=100)
        metadata = env.update_metadata(metadata={"model_dir": model_dir})
        metadata = agent.update_metadata(metadata=metadata)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    elif step == "predict":
        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)
        env = load_env(metadata)
        agent = load_agent(metadata)
        state = env.reset()
        while True:
            # steps += 1
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            env.render()
            # if done or steps > 10:
            if done:
                break


def main():
    """Main function."""
    cmdline_args = create_argument_parser().parse_args()
    env_name = cmdline_args.example
    step = cmdline_args.step
    model_dir = os.path.abspath(cmdline_args.model_dir)
    if not os.path.isdir(model_dir):
        if step == "train":
            os.makedirs(model_dir)
        else:
            raise IOError("Can not load model from a non-existent directory.")
    solve_env(env_name, step, model_dir)


if __name__ == '__main__':
    main()
