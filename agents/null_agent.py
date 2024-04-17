"""Null Agent.

An agent which does nothing.
"""
import numpy as np

from agents import BaseAgent


class NullAgent(BaseAgent):

    def take_action(self, state: tuple[int, int]) -> int:
        return 4
    
    def update(self, state: tuple[int, int], reward: float, action):
        pass