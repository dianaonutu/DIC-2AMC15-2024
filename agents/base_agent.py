"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    def __init__(self):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """

    @abstractmethod
    def take_action(self, observation: np.ndarray, info: None | dict) -> int:
        """Any code that does the action should be included here.

        Args:
            observation: The observation which is returned by the environment.
            info: Any additional information your agent needs can be passed
                in here as well as a dictionary.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, observation: np.ndarray, reward: float, action: int):
        """Any code that processes a reward given the observation is here.

        Args:
            observation: The observation which is returned by the environment.
            reward: The float value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        raise NotImplementedError
