from dataclasses import dataclass
from typing import List, Tuple
from simworlds.simworld import SimWorld, Action, State
from config import Config
from random import randint
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GamblerWorldAction(Action):
    units: int

    def __hash__(self):
        return hash(repr(self))


@dataclass
class GamblerWorldState(State):
    units: int

    def as_one_hot(self):
        units_one_hot = np.zeros(
            Config.GamblerWorldConfig.ONE_HOT_LENGTH, dtype=np.float32
        )
        units_one_hot[self.units] = 1
        return units_one_hot

    def __hash__(self):
        return hash(repr(self))


class GamblerWorld(SimWorld):
    def __init__(self):
        self.pw = Config.GamblerWorldConfig.WIN_PROBABILITY * 100
        self.max_units = 100
        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.state = GamblerWorldState(False, randint(1, 99))

    def __get_reward(self) -> int:
        """Get the reward for the current state. Basically just punish for losing and rewarding for winning, else nothing.


        Returns:
            int: Reward for the current state.
        """
        if self.state.units == 0:
            return -1000
        elif self.state.units == 100:
            return 1000
        else:
            return 0

    def get_legal_actions(self) -> List[GamblerWorldAction]:
        """Get all legal actions for the current state.
        The rule is that the gambler can only bet up to his current money.

        Returns:
            List[GamblerWorldAction]: All possible actions in the current state.
        """
        return list(
            GamblerWorldAction(i)
            for i in range(
                1, min(self.max_units - self.state.units, self.state.units) + 1
            )
        )

    def get_state(self) -> GamblerWorldState:
        """Get the current state.

        Returns:
            GamblerWorldState: Current state.
        """
        return self.state

    def do_action(self, action: GamblerWorldAction) -> Tuple[GamblerWorldState, int]:
        """Do an action (a bet).

        Args:
            action (GamblerWorldAction): The action to perform (the bet).

        Returns:
            Tuple[GamblerWorldState, int]: New state and its reward.
        """

        # A successful bet is done by sampling a random number in range [1, 100]
        # If it is less than the probability of success pw ([0, 100]) then it is a success.
        random_number = randint(1, 100)
        if random_number < self.pw:
            new_units = self.state.units + action.units
            self.state = GamblerWorldState(
                new_units == 100 or new_units == 0, new_units
            )
        else:
            new_units = self.state.units - action.units
            self.state = GamblerWorldState(
                new_units == 100 or new_units == 0, new_units
            )
        reward = self.__get_reward()
        return (self.state, reward)

