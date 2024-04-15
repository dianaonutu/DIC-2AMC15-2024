"""Null Agent.

An agent which does nothing.
"""
import numpy as np

from agents import BaseAgent


class NullAgent(BaseAgent):
    def update(self, observation: np.ndarray, reward: float, action):
        pass

    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        return 4