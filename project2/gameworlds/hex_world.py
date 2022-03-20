from distutils.command.config import config
from sklearn import neighbors
from gameworlds.gameworld import GameWorld, State, Action
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from config import Config


@staticmethod
def generate_neighbors(K=Config.k) -> dict:
    neighbors = {}
    for row in range(K):
        for col in range(K):

            if row == 0:  # Top row
                if col == 0:  # Top left corner
                    neighbors[(row, col)] = [(row, col+1), (row + 1, col)]
                elif col == K-1:  # Top right corner
                    neighbors[(row, col)] = [(row, col-1), (row + 1, col)]
                else:  # Middle pieces
                    neighbors[(row, col)] = [(row, col+1), (row, col-1),
                                             (row + 1, col), (row + 1, col - 1)]
            elif row == K-1:  # Bottom row
                if col == 0:  # Bottom left corner
                    neighbors[(row, col)] = [(row, col+1),
                                             (row - 1, col), (row - 1, col+1)]
                elif col == K-1:  # Bottom right corner
                    neighbors[(row, col)] = [(row, col-1), (row - 1, col)]
                else:  # Middle pieces
                    neighbors[(row, col)] = [(row, col+1), (row, col-1),
                                             (row - 1, col), (row - 1, col + 1)]

            else:  # Middle rows

                if col == 0:  # Left column
                    neighbors[(row, col)] = [(row, col+1), (row - 1,
                                                            col), (row + 1, col), (row - 1, col+1)]
                elif col == K-1:  # Right column
                    neighbors[(row, col)] = [(row, col-1),
                                             (row+1, col-1), (row + 1, col), (row - 1, col)]
                else:  # Middle pieces
                    neighbors[(row, col)] = [(row, col-1), (row+1, col-1),
                                             (row + 1, col), (row - 1, col), (row, col+1), (row-1, col+1)]


neighbors = generate_neighbors()


@dataclass
class HexAction(Action):

    # TODO: May incresase performance to use 1D representation and just use one index as action
    row: int
    col: int

    def __hash__(self):
        return hash(repr(self))


@dataclass
class HexState(State):

    player: int  # The player whos turn it is to move
    board: np.ndarray  # 2D repesentation of the board

    @staticmethod
    def from_array(array: List) -> State:
        """
        Convert online game representation to HexState. 
        Online format looks like this:
        Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
        """

        player = array[0]

        if player == 1:
            player = 1
        else:
            player = -1

        n = np.sqrt(len(array)-1)

        board = np.array(array[1:]).reshape((n, n))

        # TODO: Calculate final state
        final = False

        return HexState(is_final_state=final, player=player, board=board)

        # TODO? (performance gains): Make converter directly from "Flattened Game State" to ANET input without proxy through HexState

    def get_legal_actions(self):
        pass

    def do_action(self, action: HexAction) -> State:

        board = self.board.copy()

        board[action.row][action.col] = self.player

        # TODO: Calculate final state
        final = False

        return HexState(is_final_state=final, player=-self.player, board=board)

    def as_vector(self):
        # TODO: One hot-ish encode each position on board

        if self.player == 1:
            return self.board.flatten()
        return -self.board.T.flatten()  # Transpose and swap colors

    def __hash__(self):
        return hash(repr(self))


class HexWorld(GameWorld):
    # TODO: Scrap this?

    def __init__(self):
        pass
