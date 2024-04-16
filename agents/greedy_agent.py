"""Greedy Agent.

Chooses the best scoring value with no thought about the future.
"""
import numpy as np
from random import randint
from agents import BaseAgent


class GreedyAgent(BaseAgent):
    def __init__(self):
        """Chooses an action randomly unless there is a target neighboring.
        """

    def update(self, observation: np.ndarray, reward: float, action: int):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        x, y = info["agent_pos"]
        # Check each neighboring cell if there is any dirt there
        if observation[x, y + 1] == 3:
            return 0
        elif observation[x, y - 1] == 3:
            return 1
        elif observation[x - 1, y] == 3:
            return 2
        elif observation[x + 1, y] == 3:
            return 3
        else:
            return randint(0, 3)