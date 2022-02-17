from dataclasses import dataclass
from typing import List, Tuple
from simworlds.simworld import SimWorld, Action, State
from config import Config
import matplotlib.pyplot as plt
from random import uniform
from math import sin, cos, exp
import numpy as np
import itertools


@dataclass
class PoleWorldAction(Action):
    F: float

    def __hash__(self):
        return hash(repr(self))


@dataclass
class PoleWorldState(State):
    x: int
    theta: int
    dx: int
    dtheta: int

    def as_vector(self) -> np.ndarray:
        return np.array([self.x, self.theta, self.dx, self.dtheta], dtype=np.float32)

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
        self.x_max = 2.4
        self.horizontal_borders: List[float] = [-self.x_max, self.x_max]  # [m]
        # [rad]
        self.vertical_borders: List[float] = [-self.theta_max, self.theta_max]

        self.F = 10  # [N]
        self.T = 300  # Number of timesteps
        self.t = 1  # Current timestep
        self.dt = Config.PoleWorldConfig.TIMESTEP  # [s]

        self.N = Config.PoleWorldConfig.DISCRETIZATION

        self.x_discret = np.linspace(
            self.horizontal_borders[0], self.horizontal_borders[1], self.N
        )
        self.theta_discret = np.linspace(
            self.vertical_borders[0], self.vertical_borders[1], self.N
        )
        self.dx_discret = np.linspace(-5, 5, self.N)
        self.dtheta_discret = np.linspace(-5, 5, self.N)

        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.state = InternalState(0, 0, 0, uniform(-0.21, 0.21), 0, 0)
        self.external_state = self.convert_internal_to_external_state(False)
        self.pole_positions = [self.state.theta]
        self.cart_positions = [self.state.x]
        self.t = 1
        self.success = False

    def __update_state(self, F: float):
        """Update the world state according to the state equations given to us.
        """
        ddtheta = (
            self.g * sin(self.state.theta)
            + cos(self.state.theta)
            * (
                -F
                - self.m_p
                * self.L
                * (self.state.dtheta ** 2)
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

    def __within_limits(self) -> bool:
        """Check if cart and pole is within their limits.

        Returns:
            bool: If the cart and pole is within their respective limits.
        """
        return (
            self.state.theta >= -self.theta_max
            and self.state.theta <= self.theta_max
            and self.state.x >= -self.x_max
            and self.state.x <= self.x_max
        )

    def __is_final_state(self) -> Tuple[bool, bool]:
        """Check if final state and if success or fail.

        Returns:
            Tuple[bool, bool]: If it is the final state and whether it was a success/fail.
        """
        if self.t == self.T and self.__within_limits():
            self.success = True
            return True, True
        elif not self.__within_limits():
            return True, False
        else:
            return False, False

    def __get_reward(self, final_state: bool, success: bool) -> int:
        """Get the reward for this state.

        Args:
            final_state (bool): If final state
            success (bool): If success or fail

        Returns:
            int: Reward based on final state, success/fail and current internal state.
        """
        if final_state and success:
            return 1000
        elif final_state and not success:
            return -500
        return (
            1 - abs(self.state.theta) / self.theta_max -
            abs(self.state.x) / self.x_max
        )

    def get_legal_actions(self) -> List[PoleWorldAction]:
        """Get the two legal actions.

        Returns:
            List[PoleWorldAction]: The two legal actions.
        """
        return [PoleWorldAction(self.F), PoleWorldAction(-self.F)]

    def do_action(self, action: PoleWorldAction) -> Tuple[PoleWorldState, int]:
        """Do an a action and return the new state with reward.

        Args:
            action (PoleWorldAction): Action to perform.

        Returns:
            Tuple[PoleWorldState, int]: New state and reward.
        """
        self.t += 1
        self.__update_state(action.F)
        final_state, success = self.__is_final_state()
        self.external_state = self.convert_internal_to_external_state(
            final_state)
        reward = self.__get_reward(final_state, success)
        self.pole_positions.append(self.state.theta)
        self.cart_positions.append(self.state.x)

        return (self.external_state, reward)

    def convert_internal_to_external_state(self, final_state: bool) -> PoleWorldState:
        """
        Discretizes the current internal state. For example 0.21 -> 8. The integer here represents an index.
        """
        # For the neural network critic this step could possibly be bypassed.
        x_d = PoleWorld.find_nearest(self.x_discret, self.state.x)
        theta_d = PoleWorld.find_nearest(self.theta_discret, self.state.theta)
        dx_d = PoleWorld.find_nearest(self.dx_discret, self.state.dx)
        dtheta_d = PoleWorld.find_nearest(
            self.dtheta_discret, self.state.dtheta)
        return PoleWorldState(final_state, x_d, theta_d, dx_d, dtheta_d)

    def get_state(self) -> PoleWorldState:
        return self.external_state

    @staticmethod
    def find_nearest(array, value):
        """Find the index of the closest value in the array."""
        idx = (np.abs(array - value)).argmin()
        return idx
