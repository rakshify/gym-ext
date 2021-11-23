"""Implements base neural agents."""

from typing import Any, Dict

from agents.base_agents.value_agents import ModelFreeAgent
from gym_ext.envs import Env
from models import get_model_by_name


class DQNAgent(ModelFreeAgent):
    """Base class for all dqn agents."""

    name = ""

    def __init__(self, env: Env, model: str = "dqn", verbose: bool = False):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            model (str): The model to use. Fixed to dqn model
            verbose (bool): Whether to print out information.
        """
        super(DQNAgent, self).__init__(env, policy, model, verbose)
        self.target_model = get_model_by_name(model)()
        self.train_after = 1000
        model = self._get_model()
        self.model.init_vars(model=model)
        self.target_model.init_vars(model=model)
        self.batch_size = 32

    def transfer_model_weights(self):
        """Transfer the model weights to the target model."""
        self.target_model.update_weights_from_model(self.model)

    def train(self, num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        discount_factor = kwargs.get("discount_factor", 1.0)
        batch = []
        for i in range(num_episodes):
            st = time.time()
            state = self.env.reset()
            while True:
                qval, action = self.get_qval_action(state)
                state_, reward, done, _ = self.env.step(action)
                r = reward if not done else -100
                # Get next best qval from target model
                next_qvals = self.target_model.predict(state_)
                max_next_qval = np.max(next_qvals)
                y = r + discount_factor * max_next_qval * np.invert(done)
                qval[action] = y
                batch.append((state, qval))
                if len(batch) >= self.train_after:
                    b = random.sample(batch, min(len(batch), self.batch_size))
                    X, Y = zip(*b)
                    self.model.train(np.array(X), np.array(Y))
                if i % 100 == 0:
                    self.env.render()
                self.policy.exploit()
                if done:
                    self.transfer_model_weights()
                    break
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms.")
            print(msg)
            print(f"Data size till now = {len(batch)}")
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def _get_model(self):
        """Get the neural network to use for training."""
        raise NotImplementedError("Base DQN agent can not make a model.")
