"""Implements base actor-critic agents."""

import os
import time

from typing import Any, Dict

import numpy as np

from agents.base.agent import Agent
from agents.base.policy_agents import PolicyAgent
from agents.base.value_agents import ValueAgent
from gym_ext.envs import Env


class ActorCriticAgent(Agent):
    """Base class for actor-critic agents."""

    name = ""

    def set_actor_critic(self, actor: PolicyAgent, critic: ValueAgent):
        """
        Set the actor and critic.

        Args:
            actor (PolicyAgent): The actor.
            critic (ValueAgent): The critic.
        """
        self.actor = actor
        self.critic = critic

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        return self.actor.get_action(state)

    def train(self, num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            num_episodes (int): Number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        df = kwargs.get("discount_factor", 1.0)
        self.actor.policy.init_vars(self.n_features, self.n_actions)
        self.critic.model.init_vars(self.n_features, self.n_actions)
        steps = []
        last_info = 0
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()
            action = self.actor.get_action(state)
            alpha = 0.0
            self.critic.reset_vars()
            updates = []
            while True:
                alpha += 1.0
                qvals = self.critic.get_qvals(state)
                qval = qvals[action]
                # print("+" * 80)
                # print(self.critic.model.w)
                # print("+" * 80)
                # print(state)
                # print("+" * 80)
                # print(qvals)
                # print("+" * 80)
                state_, reward, done, info = self.env.step(action)
                r = reward
                if done:
                    if alpha > 170:
                        r = 5
                    else:
                        r = -10
                action_ = self.actor.get_action(state_)
                nqvals = self.critic.get_qvals(state_)
                nqval = nqvals[action_]
                agrad = self.actor.policy.grad(state, action)
                updates.append(agrad * qval / alpha)
                cgrad = self.critic.model.grad(state, action)
                update = self.critic.algorithm.get_update(
                    r, qval, nqval, cgrad, self.critic.vec_shape, df)
                self.critic.update_model(update  / alpha)
                state, action = state_, action_
                if done:
                    break
            for update in updates:
                # print(update)
                self.actor.policy.update_policy(update)
            steps.append(alpha)
            if (i + 1) % self.info_episode == 0:
                msg = (f"Finished episode {i} in "
                       f"{int((time.time() - st) * 100000) / 100}ms.")
                mx = max(steps)
                avg = int(100 * sum(steps) / len(steps)) / 100
                last = steps[last_info:]
                mx_since_last = max(last)
                avg_since_last = int(100 * sum(last) / len(last)) / 100
                print(f"Max steps so far = {mx}..."
                      f"Average steps so far = {avg}")
                print(f"Max steps since last info = {mx_since_last}..."
                      f"Average steps since last info = {avg_since_last}")
                print(msg)
                last_info = i
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(ActorCriticAgent, self).update_metadata(metadata)
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        actor_model_dir = os.path.join(model_dir, "actor")
        if not os.path.isdir(actor_model_dir):
            os.makedirs(actor_model_dir)
        critic_model_dir = os.path.join(model_dir, "critic")
        if not os.path.isdir(critic_model_dir):
            os.makedirs(critic_model_dir)
        metadata["agent"]["actor_meta"] = self.actor.update_metadata(
            {"model_dir": actor_model_dir})
        metadata["agent"]["critic_meta"] = self.critic.update_metadata(
            {"model_dir": critic_model_dir})
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env
                       ) -> "ActorCriticAgent":
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        # agent = cls(env, metadata["policy"]["name"],
        #             verbose=metadata["verbose"])
        # agent.policy.deserialize(metadata["policy"])
        return cls(env)