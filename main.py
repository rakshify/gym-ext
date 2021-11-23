"""Example runs to run the environment using solvers."""

import argparse
import json
import os

from agents import get_agent_by_name, load_agent
from gym_ext import gym, load_env


def create_argument_parser():
    """Create the argument parser."""
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
        # agent = get_agent_by_name("DQNGrid")(
        #     env, policy="greedy")
        # agent.train(num_episodes=100000)
        agent = get_agent_by_name("ContGridWorld")(
            env, model="linear", algorithm="sarsa_lambda")
        agent.train(num_episodes=10000)
        # agent = get_agent_by_name("SGDPolicyGrid")(env, policy="softmax")
        # metadata = env.update_metadata(metadata={"model_dir": model_dir})
        # metadata = agent.update_metadata(metadata=metadata)
        metadata = agent.update_metadata(metadata={"model_dir": model_dir})
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    elif step == "predict":
        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)
        # env = load_env(metadata)
        env = gym.make(env_name)
        agent = load_agent(metadata, env)
        state = env.reset()
        print(state)
        steps = 0
        while True:
            steps += 1
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            env.render()
            # if done or steps > 10:
            if done:
                break
        print(steps)


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
