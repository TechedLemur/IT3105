from typing import List, Tuple

from scipy.fftpack import tilbert
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
    tile_encoding: np.array

    def __hash__(self):
        return hash(repr(self))


@dataclass
class InternalState:
    theta1: float = 0
    dtheta1: float = 0
    theta2: float = 0
    dtheta2: float = 0
    y_tip: float = -2

    def as_vector(self) -> np.array:
        return np.array([self.theta1, self.dtheta1, self.theta2, self.dtheta2])


class SimWorld:
    def __init__(self):
        self.cfg = Config.SimWorldConfig()

        buckets = Config.TileEncodingConfig.buckets
        self.bucket_list = np.arange(
            0, buckets ** 4).reshape((buckets,) * buckets)

        theta1 = np.linspace(0, 360, buckets + 1)
        dtheta1 = np.linspace(0, 180, buckets + 1)
        theta2 = np.linspace(0, 360, buckets + 1)
        dtheta2 = np.linspace(0, 360, buckets + 1)
        tile1 = np.vstack((theta1, dtheta1, theta2, dtheta2)).T
        tile2 = tile1 + 20
        tile3 = tile2 + 20
        self.tiles = [tile1, tile2, tile3]

        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.state = InternalState()
        self.external_state = self.convert_internal_to_external_state()
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

        yp1 = 0
        yp2 = yp1 - self.cfg.L1 * np.cos(theta1)
        y_tip = yp2 - self.cfg.L2 * np.cos(theta1+theta2)

        self.state = InternalState(theta1, dtheta1, theta2, dtheta2, y_tip)

    def __is_final_state(self) -> bool:
        """Check if final state

        Returns:
            bool: If it is the final state.
        """
        return self.state.y_tip >= self.cfg.L2

    def __get_reward(self, final_state: bool) -> int:
        """Get the reward for this state.

        Args:
            final_state (bool): If final state
            success (bool): If success or fail

        Returns:
            int: Reward based on final state, and current internal state.
            -1 if not final state, 0 is final state
        """
        return - int(final_state)

    def get_legal_actions(self) -> np.array:
        """Get the two legal actions.

        Returns:
            List[PoleWorldAction]: The two legal actions.
        """
        return np.array([1.0, -1.0, 0.0])

    def do_action(self, A: int) -> Tuple[SimWorldState, int]:
        """Do an a action and return the new state with reward.

        Args:
            action (int): Action to perform.  [0,1,2] correspons to forces [-1,0,1]

        Returns:
            Tuple[PoleWorldState, int]: New state and reward.
        """
        F = A-1
        self.t += 1
        self.__update_state(F)
        final_state = self.__is_final_state()
        self.external_state = self.convert_internal_to_external_state(
            final_state)
        reward = self.__get_reward(final_state)

        return (self.external_state, reward, final_state)

    def convert_internal_to_external_state(self) -> SimWorldState:
        """
        """
        tile_encoding = np.array([])
        for tile in self.tiles:
            bucket_nr = np.argmax(
                tile - self.state.as_vector() >= 0, axis=0) - 1
            n = np.zeros(Config.TileEncodingConfig.buckets ** 4)
            n[bucket_nr] = 1
            # print(tile_encoding)
            # print(n)
            tile_encoding = np.concatenate((tile_encoding, n))
        # print(tile_encoding)
        return SimWorldState(tile_encoding)

    def get_state(self) -> SimWorldState:
        return self.external_state
