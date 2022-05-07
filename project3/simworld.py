from typing import List, Tuple

from scipy.fftpack import tilbert
from config import Config
from dataclasses import dataclass
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
    yp2: float = -1
    y_tip: float = -2
    xp2: float = 0
    x_tip: float = 0

    def as_vector(self) -> np.array:
        return np.array([self.theta1, self.dtheta1, self.theta2, self.dtheta2])

    def all(self):
        return np.array([self.theta1, self.dtheta1, self.theta2, self.dtheta2, self.y_tip])


class SimWorld:
    def __init__(self):
        self.cfg = Config.SimWorldConfig()

        buckets = Config.TileEncodingConfig.buckets
        no_tiles = Config.TileEncodingConfig.tiles
        # self.bucket_list = np.arange(0, buckets ** 4).reshape((buckets,) * 4)

        theta1 = np.append(np.linspace(-pi*2/buckets, pi, buckets-1), 2*pi)
        dtheta1 = np.append(np.linspace(-2*pi, 2*pi, buckets-1), 4*pi)
        theta2 = np.append(np.linspace(-pi*2/buckets, pi, buckets-1), 2*pi)
        dtheta2 = np.append(np.linspace(-2*pi, 2*pi, buckets-1), 10*pi)

        tiles = []
        for i in range(no_tiles):

            tile = np.vstack((theta1, dtheta1, theta2, dtheta2)).T
            tile += np.array([i*pi*2/(buckets*no_tiles), i*pi /
                             no_tiles, i*pi*2/(buckets*no_tiles), i*pi/no_tiles])

            tiles.append(tile)
        self.tiles = tiles

        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.state = InternalState()
        self.external_state = self.convert_internal_to_external_state()
        self.t = 1
        self.over = False
        self.x_history = []
        self.y_history = []
        self.internal_history = []

    def __update_state(self, F: float):
        """Update the world state according to the state equations given to us.
        """

        for _ in range(self.cfg.n):  # Apply force F for n timesteps
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
                * (
                    self.cfg.Lc2 ** 2
                    + self.cfg.L1 * self.cfg.Lc2 * cos(self.state.theta2)
                )
                + 1
            )
            d1 = (
                self.cfg.m1 * self.cfg.Lc1 ** 2
                + self.cfg.m2
                * (
                    self.cfg.L1 ** 2
                    + self.cfg.Lc2 ** 2
                    + 2 * self.cfg.L1 * self.cfg.Lc2 * cos(self.state.theta2)
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

            dtheta1 = self.state.dtheta1 + ddtheta1 * self.cfg.dt

            dtheta1 = max(min(dtheta1, 4*pi), -4*pi)

            theta1 = self.state.theta1 + dtheta1 * self.cfg.dt
            theta1 = ((theta1 + pi) % (2*pi)) - pi

            dtheta2 = self.state.dtheta2 + ddtheta2 * self.cfg.dt
            dtheta2 = max(min(dtheta2, 10*pi), -10*pi)

            theta2 = self.state.theta2 + dtheta2 * self.cfg.dt
            theta2 = ((theta2 + pi) % (2*pi)) - pi

            yp1 = 0
            yp2 = yp1 - self.cfg.L1 * cos(theta1)
            y_tip = yp2 - self.cfg.L2 * cos(theta1 + theta2)
            xp1 = 0
            xp2 = xp1 + self.cfg.L1 * sin(theta1)
            x_tip = xp2 + self.cfg.L2 * sin(theta1 + theta2)

            self.state = InternalState(
                theta1, dtheta1, theta2, dtheta2, yp2, y_tip, xp2, x_tip
            )

            x = [0, xp2, x_tip]
            y = [0, yp2, y_tip]

            self.over = (y_tip >= self.cfg.L2) or self.over

            self.x_history.append(np.array(x))
            self.y_history.append(np.array(y))
            self.internal_history.append(self.state.all())

    def __is_final_state(self) -> bool:
        """Check if final state

        Returns:
            bool: If it is the final state.
        """
        return self.over
        # return self.state.y_tip >= self.cfg.L2

    def __get_reward(self, final_state: bool) -> int:
        """Get the reward for this state.

        Args:
            final_state (bool): If final state
            success (bool): If success or fail

        Returns:
            int: Reward based on final state, and current internal state.
            -1 if not final state, 6 if final state
        """
        return 1 * int(final_state)-1

    def get_legal_actions(self) -> np.array:
        """Get the two legal actions.

        Returns:
            List[int]: The legal actions.
        """
        return np.arange(3)

    def do_action(self, A: int) -> Tuple[SimWorldState, int]:
        """Do an a action and return the new state with reward.

        Args:
            action (int): Action to perform.  [0,1,2] correspons to forces [-1,0,1]

        Returns:
            Tuple[PoleWorldState, int]: New state and reward.
        """
        F = A - 1
        self.t += 1
        self.__update_state(F)
        final_state = self.__is_final_state()
        self.external_state = self.convert_internal_to_external_state()
        reward = self.__get_reward(final_state)

        return (self.external_state, reward, final_state)

    def convert_internal_to_external_state(self) -> SimWorldState:
        """
        """
        tile_encoding = np.array([])
        for tile in self.tiles:
            bucket_nr = np.argmax(
                tile - self.state.as_vector() >= 0, axis=0)
            n = np.zeros((Config.TileEncodingConfig.buckets,) * 4)
            n[bucket_nr[0], bucket_nr[1], bucket_nr[2], bucket_nr[3]] = 1
            # print(tile_encoding)
            # print(n)
            tile_encoding = np.concatenate((tile_encoding, n.flatten()))
        # print(tile_encoding)
        return SimWorldState(tile_encoding)

    def get_state(self) -> SimWorldState:
        return self.external_state

    def plot_current_state(self):
        x = [0, self.state.xp2, self.state.x_tip]
        y = [0, self.state.yp2, self.state.y_tip]
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, "o-", lw=2)
        plt.xlim(-2, 2)
        plt.ylim(-2.5, 1.5)
        plt.show()
