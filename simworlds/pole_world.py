from typing import List
from simworlds.simworld import SimWorld, Action
from config import Config
from random import randint, random
import matplotlib.pyplot as plt


class PoleWorldAction(Action):
    def __init__(self, units_to_bet: int):
        self.units = units_to_bet

    def __repr__(self):
        return str(self.units)


class PoleWorld(SimWorld):
    def __init__(self):
        self.L = 0.5  # [m]
        self.m_p = 0.1  # [kg]
        self.m_c = 1.0  # [kg]
        self.g = 9.8  # [m/s^2]
        self.theta = 1  # [rad]

    def get_legal_actions(self) -> List[PoleWorldAction]:
        pass

    def do_action(self, action: PoleWorldAction) -> bool:
        pass

    def get_state(self):
        pass

    def visualize_individual_solutions(self):
        plt.plot(range(1, 100), range(1, 100))
        plt.xlabel("State")
        plt.ylabel("Wager")
        plt.show()

    def visualize_learning_progress(self):
        pass

    def __check_legal_action(self, action: PoleWorldAction):
        pass
