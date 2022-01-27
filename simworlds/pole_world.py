from dataclasses import dataclass
from typing import List, Tuple
from simworlds.simworld import SimWorld, Action, State
from config import Config
import matplotlib.pyplot as plt
from random import uniform
from math import sin, cos, exp


@dataclass
class PoleWorldAction(Action):
    F: float

    def __hash__(self):
        return hash(repr(self))


@dataclass
class PoleWorldState(State):
    def __hash__(self):
        return hash(repr(self))


@dataclass
class InternalState:
    x: float
    dx: float
    ddx: float
    theta: float
    dtheta: float
    ddtheta: float


class PoleWorld(SimWorld):
    def __init__(self):
        self.L = Config.PoleWorldConfig.POLE_LENGTH  # [m]
        self.m_p = Config.PoleWorldConfig.POLE_MASS  # [kg]
        self.m_c = 1.0  # [kg]
        self.g = Config.PoleWorldConfig.GRAVITY  # [m/s^2]

        self.theta_max = 0.21  # [rad]
        self.horizontal_borders: List[float] = [-2.4, 2.4]  # [m]
        self.vertical_borders: List[float] = [-self.theta_max, self.theta_max]  # [rad]

        self.F = 10  # [N]
        self.T = 300  # Number of timesteps
        self.dt = Config.PoleWorldConfig.TIMESTEP  # [s]

        self.state = PoleWorldState(0, 0, 0, uniform(-0.21, 0.21), 0, 0)

    def __update_state(self, F: float):
        ddtheta = (
            self.g * sin(self.state.theta)
            + cos(self.state.theta)
            * (
                -F
                - self.m_p
                * self.L
                * self.state.ddtheta
                * sin(self.state.theta)
                / (self.m_p + self.m_c)
            )
        ) / (
            self.L
            * (4 / 3 - (self.m_p * cos(self.state.theta) ** 2) / (self.m_c + self.m_p))
        )
        ddx = (
            F
            + self.m_p
            * self.L
            * (
                self.state.dtheta ** 2 * sin(self.state.theta)
                - self.state.ddtheta * cos(self.state.theta)
            )
        ) / (self.m_c + self.m_p)

        dtheta = self.state.dtheta + ddtheta * self.dt
        dx = self.state.dx + ddx * self.dt
        theta = self.state.theta + dtheta * self.dt
        x = self.state.x + dx * self.dt

        self.state = InternalState(x, dx, ddx, theta, dtheta, ddtheta)
        self.external_state = PoleWorldState(False)

    def __get_reward(self) -> int:
        return exp(-((self.theta_max * self.state.theta) ** 2))

    def get_legal_actions(self) -> List[PoleWorldAction]:
        return [PoleWorldAction(self.F), PoleWorldAction(-self.F)]

    def do_action(self, action: PoleWorldAction) -> Tuple[PoleWorldState, int]:
        self.__update_state(action.F)
        self.external_state = PoleWorldState(False)
        reward = self.__get_reward()
        return (self.external_state, reward)

    def get_state(self) -> PoleWorldState:
        return self.external_state

    def visualize_individual_solutions(self):
        plt.plot(range(1, 100), range(1, 100))
        plt.xlabel("State")
        plt.ylabel("Wager")
        plt.show()

    def visualize_learning_progress(self):
        print("ok")
