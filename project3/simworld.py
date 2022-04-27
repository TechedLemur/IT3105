from typing import List, Tuple
from config import Config
from dataclasses import dataclass
import numpy as np
from math import sin, cos, pi


@dataclass
class SimWorldAction:
    F: float

    def __hash__(self):
        return hash(repr(self))


@dataclass
class SimWorldState:
    theta1: int
    dtheta1: int
    theta2: int
    dtheta2: int

    def as_tile_encoding(self) -> np.ndarray:
        return np.array(
            [self.theta1, self.dtheta1, self.theta2, self.dtheta2], dtype=np.float32
        )

    def __hash__(self):
        return hash(repr(self))


@dataclass
class InternalState:
    theta1: float
    dtheta1: float
    theta2: float
    dtheta2: float


class SimWorld:
    def __init__(self):
        self.cfg = Config.SimWorldConfig()
        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.state = InternalState()
        self.external_state = self.convert_internal_to_external_state(False)
        self.t = 1
        self.success = False

    def __update_state(self, F: float):
        """Update the world state according to the state equations given to us.
        """
        phi2 = (
            self.cfg.m2
            * self.cfg.Lc2
            * self.cfg.g
            * cos(self.state.theta1 + self.state.theta2 - pi / 2)
        )
        phi1 = (
            -self.cfg.m2
            * self.cfg.L1
            * self.cfg.Lc2
            * self.state.dtheta2 ** 2
            * sin(self.state.theta2)
            - 2
            * self.cfg.m2
            * self.cfg.L1
            * self.cfg.Lc2
            * self.state.dtheta1
            * self.state.dtheta2
            * sin(self.state.theta2)
            + (self.cfg.m1 * self.cfg.Lc1 + self.cfg.m2 * self.cfg.L1)
            * self.cfg.g
            * cos(self.state.theta1 - pi / 2)
            + phi2
        )

        d2 = (
            self.cfg.m2
            * (self.cfg.Lc2 ** 2 + self.cfg.L1 * self.cfg.Lc2 * cos(self.state.theta2))
            + 1
        )
        d1 = (
            self.cfg.m1 * self.cfg.Lc1 ** 2
            + self.cfg.m2
            * (
                self.cfg.L1 ** 2
                + self.cfg.Lc2 ** 2
                + 2 * self.cfg.L1 * self.cfg.Lc2 * cos(self.state.theta)
            )
            + 2
        )

        ddtheta2 = (
            F
            + d2 / d1 * phi1
            - self.cfg.m2
            * self.cfg.L1
            * self.cfg.Lc2
            * self.state.dtheta1 ** 2
            * sin(self.state.theta2)
            - phi2
        ) / (self.cfg.m2 * self.cfg.Lc2 ** 2 + 1 - (d2 ** 2) / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dtheta1 = self.state.dtheta1 + ddtheta1 * self.dt
        theta1 = self.state.theta1 + dtheta1 * self.dt
        dtheta2 = self.state.dtheta2 + ddtheta2 * self.dt
        theta2 = self.state.theta + dtheta2 * self.dt

        self.state = InternalState(theta1, dtheta1, theta2, dtheta2)

    def __is_final_state(self) -> bool:
        """Check if final state and if success or fail.

        Returns:
            Tuple[bool, bool]: If it is the final state and whether it was a success/fail.
        """
        return True

    def __get_reward(self, final_state: bool, success: bool) -> int:
        """Get the reward for this state.

        Args:
            final_state (bool): If final state
            success (bool): If success or fail

        Returns:
            int: Reward based on final state, success/fail and current internal state.
        """
        return 1e18

    def get_legal_actions(self) -> np.array:
        """Get the two legal actions.

        Returns:
            List[PoleWorldAction]: The two legal actions.
        """
        return np.array([1.0, -1.0, 0.0])

    def do_action(self, F: float) -> Tuple[SimWorldState, int]:
        """Do an a action and return the new state with reward.

        Args:
            action (PoleWorldAction): Action to perform.

        Returns:
            Tuple[PoleWorldState, int]: New state and reward.
        """
        self.t += 1
        self.__update_state(F)
        final_state, success = self.__is_final_state()
        self.external_state = self.convert_internal_to_external_state(final_state)
        reward = self.__get_reward(final_state, success)

        return (self.external_state, reward)

    def convert_internal_to_external_state(self, final_state: bool) -> SimWorldState:
        """
        Discretizes the current internal state. For example 0.21 -> 8. The integer here represents an index.
        """
        return False

    def get_state(self) -> SimWorldState:
        return self.external_state
