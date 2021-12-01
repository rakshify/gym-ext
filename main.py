"""Example runs to run the environment using solvers."""

import argparse
import json
import os

from agents import get_agent_by_name, load_agent
from gym_ext import gym, load_env, GYM_EXT_ENVS


def create_argument_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description='parse incoming text')
    parser.add_argument('--environment', '-e',
                        required=True,
                        help="Environment to run by name.")
    parser.add_argument('--agent', '-a',
                        help="Agent to use to solve.")
    parser.add_argument('--step', '-s',
                        required=True,
                        help="Train/Predict.")
    parser.add_argument('--model_dir', '-d',
                        required=True,
                        help="Model directory to save or load the model from.")
    return parser


def solve_env(env_name: str, agent_name: str, step: str, model_dir: str,
              **agent_kwargs):
    """Solve the environment."""
    if step == "train":
        env = gym.make(env_name)
        agent = get_agent_by_name(agent_name, env, **agent_kwargs)
        agent.train(num_episodes=agent_kwargs.get("num_episodes", 10000))
        if env_name in GYM_EXT_ENVS:
            metadata = env.update_metadata(metadata={"model_dir": model_dir})
            metadata = agent.update_metadata(metadata=metadata)
        else:
            metadata = agent.update_metadata(metadata={"model_dir": model_dir})
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    elif step == "predict":
        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)
        if env_name in GYM_EXT_ENVS:
            env = load_env(metadata)
        else:
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
    env_name = cmdline_args.environment
    agent_name = cmdline_args.agent
    step = cmdline_args.step
    model_dir = os.path.abspath(cmdline_args.model_dir)
    if step == "train" and not agent_name:
        raise ValueError("Agent name is required for training.")
    if not os.path.isdir(model_dir):
        if step == "train":
            os.makedirs(model_dir)
        else:
            raise IOError("Can not load model from a non-existent directory.")
    # agent_kwargs = {
    #     "model": "linear",
    #     "algorithm": "sarsa_lambda"
    # }
    # agent_kwargs = {"policy": "softmax", "num_episodes": 500000}
    agent_kwargs = {
        "actor": {
            "name": "ContSGDPolicyGrid",
            "policy": "softmax"
        },
        "critic": {
            "name": "ContGridWorld",
            "model": "linear",
            "algorithm": "sarsa_lambda"
        },
        "num_episodes": 10000
    }
    solve_env(env_name, agent_name, step, model_dir, **agent_kwargs)


if __name__ == '__main__':
    main()
