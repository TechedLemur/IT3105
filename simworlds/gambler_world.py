from dataclasses import dataclass
from typing import List
from simworlds.simworld import SimWorld, Action, State
from config import Config
from random import randint, random
import matplotlib.pyplot as plt


@dataclass
class GamblerWorldAction(Action):
    units: int

    def __hash__(self):
        return hash(repr(self))


@dataclass
class GamblerWorldState(State):
    state: int

    def __hash__(self):
        return hash(repr(self))


class GamblerWorld(SimWorld):
    def __init__(self):
        self.pw = Config.GamblerWorldConfig.WIN_PROBABILITY * 100
        self.max_units = 100
        self.units = GamblerWorldState(randint(1, 99))

    def get_legal_actions(self) -> List[GamblerWorldAction]:
        return list(
            GamblerWorldAction(i)
            for i in range(
                1, min(self.max_units - self.units.state, self.units.state) + 1
            )
        )

    def do_action(self, action: GamblerWorldAction) -> bool:
        if self.__check_legal_action(action):
            # A successful bet is done by sampling a random number in range [1, 100]
            # If it is less than the probability of success pw ([0, 100]) then it is a success.
            random_number = randint(1, 100)
            if random_number < self.pw:
                self.units.state += action.units
            return True
        else:
            return False

    def get_state(self) -> GamblerWorldState:
        return self.units

    def visualize_individual_solutions(self):
        plt.plot(range(1, 100), range(1, 100))
        plt.xlabel("State")
        plt.ylabel("Wager")
        plt.show()

    def visualize_learning_progress(self):
        pass

    def __check_legal_action(self, action: GamblerWorldAction):
        return (
            action.units <= self.max_units - self.units.state
            and action.units <= self.units.state
        )
