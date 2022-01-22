from typing import List
from simworlds.simworld import SimWorld, Action
from config import Config
from random import randint, random
import matplotlib.pyplot as plt


class GamblerWorldAction(Action):
    def __init__(self, units_to_bet: int):
        self.units = units_to_bet

    def __repr__(self):
        return str(self.units)


class GamblerWorld(SimWorld):
    def __init__(self):
        self.pw = Config.GamblerWorldConfig.WIN_PROBABILITY * 100
        self.max_units = 100
        self.units = randint(1, 99)

    def get_legal_actions(self) -> List[GamblerWorldAction]:
        print("self.units: ", self.units)
        return list(
            GamblerWorldAction(i)
            for i in range(1, min(self.max_units - self.units, self.units) + 1)
        )

    def do_action(self, action: GamblerWorldAction) -> bool:
        if self.__check_legal_action(action):
            random_number = randint(1, 100)
            if random_number < self.pw:
                self.units += action.units
            return True
        else:
            return False

    def get_state(self):
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
            action.units <= self.max_units - self.units and action.units <= self.units
        )
