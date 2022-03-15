from numpy import dtype
from gameworlds.gameworld import GameWorld, State, Action
from typing import List, Tuple
from dataclasses import dataclass
from config import Config
import numpy as np


@dataclass
class NimState(State):
    pieces: int

    def as_vector(self):
        one_hot = np.zeros(Config.N, dtype=np.float32)
        one_hot[self.pieces] = 1
        return one_hot

    def __hash__(self):
        return hash(repr(self))


@dataclass
class NimAction(Action):
    pieces: int

    def __hash__(self):
        return hash(repr(self))


class NimWorld(GameWorld):
    """
    A very simple version of NIM provides a nice challenge for MCTS. The 2-player game is defined by two key parameters: 
    N and K. N is the number of pieces on the board, and K is the maximum number that a player can take off the
    board on their turn; the minimum pieces to remove is always ONE
    """

    def __init__(self, K: int = Config.K, N: int = Config.N):

        self.state = NimState(is_final_state=N == 0, pieces=N)
        self.K = K
        self.all_actions = list(
            NimAction(i)
            for i in range(1, K+1)
        )

    def get_legal_actions(self):

        return list(
            NimAction(i)
            for i in range(1, min(self.K, self.state.pieces)+1)
        )

    def do_action(self, action: NimAction) -> State:

        new_amount = self.state.pieces - action.pieces

        if new_amount < 0:
            raise Exception("Illegal move")

        self.state = NimState(new_amount == 0, new_amount)
        return self.state

    def get_all_actions(self):
        return self.all_actions

    def get_state(self):
        return self.state
