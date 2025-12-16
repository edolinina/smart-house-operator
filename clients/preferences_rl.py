import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from clients.smart_device_base import SmartInput


class PreferenceEnv(gym.Env):
    """Gymnasium environment for learning individual preferences.

    Each persona has a preferred value (e.g., temperature or brightness),
    and the agent gets a reward based on how close its action is to the persona's preference.
    """

    def __init__(self, personas: list[str], preferences: dict[str, int], bounds: tuple[int, int]) -> None:
        """
        Initialize PreferenceEnv.

        Args:
            personas (list[str]): List of persona names to include in the environment.
            preferences (dict[str, int]): Mapping of persona to their preferred value.
            bounds (tuple[int, int]): Min and max possible action/state values.
        """
        super().__init__()
        self.personas = personas
        self.preferences = {k: v for k, v in preferences.items() if k in personas}
        self.min_state, self.max_state = bounds
        self.current_persona: str | None = None
        self.action_space = spaces.Discrete(self.max_state - self.min_state + 1)
        self.observation_space = spaces.Discrete(len(self.personas))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        """
        Reset environment to start a new episode.

        Args:
            seed (int | None): Random seed for reproducibility.
            options (dict | None): Optional gymnasium reset options.

        Returns:
            tuple[int, dict]: Index of selected persona and empty info dict.
        """
        super().reset(seed=seed)
        self.current_persona = random.choice(self.personas)
        return self.personas.index(self.current_persona), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """
        Take an action in the environment and return reward.

        Args:
            action (int): Index representing the chosen action.

        Returns:
            tuple[int, float, bool, bool, dict]: 
                - index of current persona,
                - reward (closer to preference -> higher),
                - terminated flag (always True),
                - truncated flag (always False),
                - info dict (empty).
        """
        action_res = self.min_state + action
        ideal_res = self.preferences[self.current_persona]
        reward = -abs(action_res - ideal_res)  # closer to ideal => higher reward
        terminated = True  # one-step episode
        return self.personas.index(self.current_persona), reward, terminated, False, {}


class PreferenceRLAgent(SmartInput):
    """Reinforcement learning agent for PreferenceEnv."""

    def __init__(self, env: PreferenceEnv, lr: float = 0.1, eps: float = 0.2, scale_column: int = 1) -> None:
        """
        Initialize RL agent with Q-table and hyperparameters.

        Args:
            env (PreferenceEnv): Environment instance.
            lr (float): Learning rate for Q-table updates.
            eps (float): Exploration probability (epsilon-greedy).
            scale_column (int): Factor to scale the column headers in returned preferences.
        """
        self.env = env
        self.q_table = np.zeros((len(env.personas), env.action_space.n))
        self.lr = lr
        self.eps = eps
        self.scale_column = scale_column

    def choose_action(self, state: int) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            state (int): Current persona index.

        Returns:
            int: Selected action index.
        """
        if random.random() < self.eps:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def train(self, episodes: int = 30, plot: bool = False) -> None:
        """
        Train the agent using Q-learning (bandit-style) and track preference error.
        """
        errors = []

        for _ in range(episodes):
            state, _ = self.env.reset()
            persona = self.env.current_persona

            action = self.choose_action(state)
            _, reward, _, _, _ = self.env.step(action)

            # Q-learning update
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.lr * (reward - old_value)

            # Compute error between learned best action and ideal preference
            best_action = np.argmax(self.q_table[state])
            ideal_action = self.env.preferences[persona] - self.env.min_state
            error = abs(best_action - ideal_action)

            errors.append(error)

        if plot:
            # Plot learning curve
            plt.figure()
            plt.plot(errors)
            plt.xlabel("Episode")
            plt.ylabel("Absolute Preference Error")
            plt.title("Learning Curve: Preference Error per Episode")
            plt.grid(True)
            plt.show()

    def train2(self, episodes: int = 200) -> None:
        """
        Train the agent using Q-learning over a number of episodes.

        Args:
            episodes (int): Number of episodes to train.
        """
        for _ in range(episodes):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            _, reward, _, _, _ = self.env.step(action)
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.lr * (reward - old_value)

        best_action = np.argmax(self.q_table[state])
        ideal_action = self.preferences[self.env.current_persona] - self.env.min_state
        error = abs(best_action - ideal_action)

    def get_state(self) -> dict[str, dict[int, float]]:
        """
        Convert Q-table to a probability distribution over possible values for each persona.

        Returns:
            dict[str, dict[int, float]]: Mapping of persona to probability distribution of actions.
        """
        col_values = list(range(self.env.min_state, self.env.max_state + 1)) 
        q_df = pd.DataFrame(self.q_table, index=self.env.personas, columns=col_values)

        # Shift Q-values to all positive
        q_shifted = q_df - q_df.min(axis=1).values[:, None] + 1e-6

        # Normalize to sum to 1 for each persona
        prob_df = q_shifted.div(q_shifted.sum(axis=1), axis=0)

        # Scale column headers to fit into the desired range
        prob_df.columns = [col * self.scale_column for col in prob_df.columns]

        return prob_df.round(2).to_dict(orient='index')
