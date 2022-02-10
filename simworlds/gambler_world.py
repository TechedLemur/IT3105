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
        units_one_hot = np.zeros(101, dtype=np.float32)
        units_one_hot[self.units] = 1
        return units_one_hot

    def __hash__(self):
        return hash(repr(self))


class GamblerWorld(SimWorld):
    def __init__(self):
        self.pw = Config.GamblerWorldConfig.WIN_PROBABILITY * 100
        self.max_units = 100
        self.state = GamblerWorldState(False, randint(1, 99))

    def __get_reward(self) -> int:
        if self.state.units == 0:
            return -1000
        elif self.state.units == 100:
            return 1000
        else:
            return 0

    def get_legal_actions(self) -> List[GamblerWorldAction]:
        return list(
            GamblerWorldAction(i)
            for i in range(
                1, min(self.max_units - self.state.units, self.state.units) + 1
            )
        )

    def do_action(self, action: GamblerWorldAction) -> Tuple[GamblerWorldState, int]:
        # A successful bet is done by sampling a random number in range [1, 100]
        # If it is less than the probability of success pw ([0, 100]) then it is a success.
        random_number = randint(1, 100)
        if random_number < self.pw:
            new_units = self.state.units + action.units
            self.state = GamblerWorldState(
                new_units == 100 or new_units == 0, new_units)
        else:
            new_units = self.state.units - action.units
            self.state = GamblerWorldState(
                new_units == 100 or new_units == 0, new_units)
        reward = self.__get_reward()
        return (self.state, reward)

    def visualize_individual_solutions(self):
        plt.plot(range(1, 100), range(1, 100))
        plt.xlabel("State")
        plt.ylabel("Wager")
        plt.show()

    def visualize_learning_progress(self):
        pass
