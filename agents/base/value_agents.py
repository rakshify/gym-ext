"""Implements base value agents."""

from typing import Any, Dict

from agents.base_agents.agent import Agent
from algorithms import get_algorithm_by_name
from gym_ext.envs import Env
from models import get_model_by_name


class ValueAgent(Agent):
    """Base class for value agents."""

    def __init__(self, env: Env, verbose: bool = False, **kwargs):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            verbose (bool): Whether to print out information.
            **kwargs: Additional arguments.
        """
        super(ValueAgent, self).__init__(env, verbose, **kwargs)
        self.epsilon = kwargs.get("epsilon", 0.99)
        self.decay = kwargs.get("decay", 0.99)
        self.max_eps = kwargs.get("max_eps", 0.99)
        self.min_eps = kwargs.get("min_eps", 0.01)

    def get_action(self, state: Any) -> Any:
        """
        Get the action to take.

        Args:
            state (Any): The state to get the action for.

        Returns:
            Any: The action to take.
        """
        q_val, action = self.get_qval_action(state)
        return action

    def get_qval_action(self, state: Union[int, np.ndarray]
                        ) -> Tuple[np.ndarray, int]:
        """
        Get the q-value and action for a state.

        Args:
            state (Union[int, np.ndarray]): The state to get the q-value
                                            and action for.

        Returns:
            Tuple[np.ndarray, int]: The q-value and action.
        """
        # TODO: Implement this method.
        raise NotImplementedError("Temporarily not implemented")

    def explore_policy(self):
        """Explore the policy."""
        self.epsilon = min(self.max_eps, self.epsilon / self.decay)

    def exploit_policy(self):
        """Exploit the policy."""
        self.epsilon = max(self.min_eps, self.epsilon * self.decay)

    def _eps_greedy(self, q_vals: np.ndarray) -> int:
        """
        Epsilon greedy.

        Args:
            q_vals (np.ndarray): The q-values.

        Returns:
            int: The action argument.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_vals.shape[0])
        else:
            return np.argmax(q_vals)

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(ValueAgent, self).update_metadata(metadata)
        metadata["agent"]["epsilon"] = self.epsilon
        return metadata


class ModelFreeValueAgent(ValueAgent):
    """Base class for all model-free value agents."""

    name = ""

    def __init__(self, env: Env, model: str, verbose: bool = False,
                 **kwargs):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            model (str): The model to use.
            verbose (bool): Whether to print out information.
            **kwargs: Additional arguments.
        """
        super(ModelFreeValueAgent, self).__init__(env, verbose, **kwargs)
        if model is not None:
            self.model_name = model
            self.model = get_model_by_name(model)()

    def get_qval_action(self, state: Union[int, np.ndarray]
                        ) -> Tuple[np.ndarray, Any]:
        """
        Get the q-value and action for a state.

        Args:
            state (Union[int, np.ndarray]): The state to get the q-value
                                            and action for.

        Returns:
            Tuple[np.ndarray, Any]: The q-value and action.
        """
        qvals = self.model.predict(state)
        return qvals, self._eps_greedy(qvals)

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(ModelFreeValueAgent, self).update_metadata(metadata)
        meta = self.model.serialize()
        model_dir = metadata.get("model_dir")
        if not os.path.isdir(model_dir):
            raise IOError(f"Model directory {model_dir} not found")
        model_dir = os.path.join(model_dir, "agent-model-vars")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        meta["vars"] = self.model.save_vars(model_dir)
        metadata["agent"]["model"] = meta
        return metadata

    def update_model(self, update: Any):
        """
        Update the model.

        Args:
            update (Any): The update to apply.
        """
        self.model.update(update)

    def q_grad(self, state: Union[int, np.ndarray], action: int) -> np.ndarray:
        """
        Get the gradient of the q-value for a state and action.

        Args:
            state (Union[int, np.ndarray]): The state to get the gradient for.
            action (int): The action to get the gradient for.

        Returns:
            np.ndarray: The gradient of the q-value for the state and action.
        """
        return self.model.grad(state, action)

    @property
    def vec_shape(self):
        """Get the shape of the model weights."""
        return self.model.vec_shape



class AlgorithmBasedAgent(ModelFreeValueAgent):
    """Base class for all algorithm-based agents."""

    name = ""

    def __init__(self, env: Env, model: str, algorithm: str,
                 verbose: bool = False, **kwargs):
        """
        Initialize the base agent.

        Args:
            env(Env): Environment for this agent
            model (str): The model to use.
            algorithm (str): The algorithm to use.
            verbose (bool): Whether to print out information.
            **kwargs: Additional arguments.
        """
        super(AlgorithmBasedAgent, self).__init__(
            env, model, verbose, **kwargs)
        self.algorithm_name = algorithm
        self.algorithm = get_algorithm_by_name(algorithm)()

    def train(self, num_episodes: int = 10000, **kwargs):
        """
        Train the agent.

        Args:
            num_episodes (int): The number of episodes to train for.
            **kwargs: Additional arguments.
        """
        start = time.time()
        discount_factor = kwargs.get("discount_factor", 1.0)
        self.model.init_vars(self.n_features, self.n_actions)
        episode_rewards = []
        for i in range(num_episodes):
            st = time.time()
            er = self.algorithm.solve_episode(self.env, self, discount_factor)
            episode_rewards.append(er)
            self.exploit_policy()
            msg = (f"Finished episode {i} in "
                   f"{int((time.time() - st) * 100000) / 100}ms "
                   f"with reward = {er}.")
            print(msg)
        print(f"Model trained in {int((time.time() - start) * 100) / 100}sec.")

    def update_metadata(self, metadata: Dict[str, Any]):
        """
        Update the metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to update.
        """
        metadata = super(AlgorithmBasedAgent, self).update_metadata(metadata)
        meta = self.algorithm.serialize()
        metadata["agent"]["algorithm"] = meta
        return metadata

    @classmethod
    def load_from_meta(cls, metadata: Dict[str, Any], env: Env
                       ) -> "AlgorithmBasedAgent":
        """
        Load the agent from metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to load from.
            env (Env): The environment to use.

        Returns:
            Agent: The loaded agent.
        """
        model = metadata["model"]
        algorithm = metadata["algorithm"]
        agent = cls(env, model["name"], algorithm["name"],
                    epsilon=metadata["epsilon"])
        agent.algorithm.load_vars(algorithm)
        agent.model.load_vars(model["vars"])
        return agent