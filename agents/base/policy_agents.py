"""Implements base policy agent."""

from typing import Any, Dict

from agents.base_agents.agent import Agent
from gym_ext.envs import Env
from policies import get_policy_by_name


class PolicyAgent(Agent):
    """Base class for policy agents."""

    name = ""

    def __init__(self, env: Env, policy: str, verbose: bool = False,
                 **kwargs):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            policy (str): The policy to use.
            verbose (bool): Whether to print out information.
            **kwargs: Additional arguments.
        """
        super(PolicyAgent, self).__init__(env, verbose, **kwargs)
        self.policy_name = policy
        self.policy = get_policy_by_name(policy)()

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata["agent"] = {
            "name": self.name,
            "policy": self.policy.serialize()
        }
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env
                       ) -> "PolicyAgent":
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        policy = metadata["policy"]
        agent = cls(env, policy["name"])
        agent.policy.load_vars(policy["vars"])
        return agent


class ModelFreePolicyAgent(PolicyAgent):
    """Base class for all model-free policy agents."""

    name = ""

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        return self.policy.get_action(state)

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(ModelFreePolicyAgent, self).update_metadata(metadata)
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        model_dir = os.path.join(model_dir, "agent-policy-vars")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        meta = {"vars": self.policy.save_vars(model_dir)}
        metadata["agent"]["policy"].update(meta)
        return metadata

    def train(self, num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        self.policy.init_vars(self.n_features, self.n_actions)
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()

            # Step 1: Collect history
            history = []
            while True:
                action = self.get_action(state)
                state_, reward, done, _ = self.env.step(action)
                # r = reward if not done else -100
                update = self.policy.grad(state, action)
                history.append((state, action, reward, update))
                state = state_
                if done:
                    break

            # Step 2: Cumulative rewards for state values
            cum_rewards = []
            cum_rewards.append(history[-1][2])
            states, actions, rewards, updates = zip(*history)
            for state, action, reward, update in reversed(history):
                cum_rewards.append(reward + cum_rewards[-1])
            cum_rewards = np.array(cum_rewards)[::-1]

            # Step 3: Update policy
            alpha = 0.0
            print(f"Steps in this episode = {len(history)}")
            history = zip(states, actions, cum_rewards, updates)
            for state, action, reward, update in history:
                alpha += 1
                self.policy.update_policy((1 / alpha) * update * reward)
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")
