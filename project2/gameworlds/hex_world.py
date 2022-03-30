from copy import copy
from gameworlds.gameworld import GameWorld, State, Action
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from config import cfg
import networkx as nx
import matplotlib.pyplot as plt
import copy
from IPython.display import clear_output


def generate_neighbors(K=cfg.k) -> dict:
    """
    Helper method for generating a look-up table for neighbouring nodes.
    Generate once and reuse to increase performance.
    """
    neighbors = {}
    for row in range(K):
        for col in range(K):

            if row == 0:  # Top row
                if col == 0:  # Top left corner
                    neighbors[(row, col)] = [(row, col + 1), (row + 1, col)]
                elif col == K - 1:  # Top right corner
                    neighbors[(row, col)] = [
                        (row, col - 1),
                        (row + 1, col),
                        (row + 1, col - 1),
                    ]
                else:  # Middle pieces
                    neighbors[(row, col)] = [
                        (row, col + 1),
                        (row, col - 1),
                        (row + 1, col),
                        (row + 1, col - 1),
                    ]
            elif row == K - 1:  # Bottom row
                if col == 0:  # Bottom left corner
                    neighbors[(row, col)] = [
                        (row, col + 1),
                        (row - 1, col),
                        (row - 1, col + 1),
                    ]
                elif col == K - 1:  # Bottom right corner
                    neighbors[(row, col)] = [(row, col - 1), (row - 1, col)]
                else:  # Middle pieces
                    neighbors[(row, col)] = [
                        (row, col + 1),
                        (row, col - 1),
                        (row - 1, col),
                        (row - 1, col + 1),
                    ]

            else:  # Middle rows

                if col == 0:  # Left column
                    neighbors[(row, col)] = [
                        (row, col + 1),
                        (row - 1, col),
                        (row + 1, col),
                        (row - 1, col + 1),
                    ]
                elif col == K - 1:  # Right column
                    neighbors[(row, col)] = [
                        (row, col - 1),
                        (row + 1, col - 1),
                        (row + 1, col),
                        (row - 1, col),
                    ]
                else:  # Middle pieces
                    neighbors[(row, col)] = [
                        (row, col - 1),
                        (row + 1, col - 1),
                        (row + 1, col),
                        (row - 1, col),
                        (row, col + 1),
                        (row - 1, col + 1),
                    ]
    return neighbors


neighbors = generate_neighbors()


def generate_bridge_dict(K=cfg.k):
    """
    Generate a dictionary mapping each cell to their possible bridge carrier and endpoints
    Format [(carrier1, carrier2, endpoind1), (carrier3, carrier4, endpoind2)...]
    """

    bridges = {}

    for row in range(K):
        for col in range(K):

            up = ((row-1, col), (row, col-1), (row-1, col-1))

            up_right = ((row+1, col-1), (row, col-1), (row+1, col-2))
            down_right = ((row+1, col-1), (row+1, col), (row+2, col-1))

            down = ((row+1, col), (row, col+1), (row+1, col+1))

            down_left = ((row, col+1), (row - 1, col+1), (row-1, col+2))
            up_left = ((row-1, col+1), (row - 1, col), (row-2, col+1))

            patterns = [up, up_right, up_left, down, down_left, down_right]

            b = []
            for pattern in patterns:
                bridge_endpoint = pattern[2]
                if bridge_endpoint[0] >= 0 and bridge_endpoint[1] >= 0 and bridge_endpoint[0] < K and bridge_endpoint[1] < K:
                    b.append(pattern)

            bridges[(row, col)] = b
    return bridges


bridges = generate_bridge_dict()


def is_final_move(move: Tuple[int, int], player=1, k=cfg.k, board: np.ndarray = None) -> bool:
    """
    Returns True if the move is a winning move for the player, False otherwise.
    Do a breadth-first search from the placed piece, and see if the two edges for the player is connected.
    Worst case O(k^2), best case O(2)
    """

    if player == 1:
        ind = 0
    else:
        ind = 1

    side1 = move[ind] == 0  # Piece inserted in top/left row
    side2 = move[ind] == k - 1  # Piece inserted in bottom/right row

    Q = list(neighbors[move])
    visited = set()
    while Q:
        p = Q.pop()
        visited.add(p)

        if board[p] == player:
            if p[ind] == 0:
                side1 = True
            elif p[ind] == k - 1:
                side2 = True

            if side1 and side2:  # Connected
                return True

            for n in neighbors[p]:
                # Possible performance gains to check if board[n]==player
                if n not in Q and n not in visited:
                    Q.append(n)

    return False


@dataclass
class HexAction(Action):

    # TODO: May incresase performance to use 1D representation and just use one index as action
    row: int
    col: int

    def transposed(self) -> Action:
        return HexAction(row=self.col, col=self.row)

    def __hash__(self):
        return hash(repr(self))


@dataclass
class HexState(State):

    player: int  # The player whos turn it is to move
    board: np.ndarray  # 2D repesentation of the board
    k: int  # Boardlength

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

        n = int(np.sqrt(len(array) - 1))

        board = np.array(array[1:]).reshape((n, n))

        board = np.where(board > 1, -1, board)

        final = False
        # Calculate final state. Maybe not necessary, as there is no use to continue after a game is finished
        # for i in range(n):
        #     if player == 1:
        #         if is_final_move(
        #             move=(0, i), player=1, k=n, board=board
        #         ):  # Check all top pieces
        #             final = True
        #             break
        #     else:  # Player -1
        #         # Check all left pieces
        #         if is_final_move(move=(i, 0), player=-1, k=n, board=board):
        #             final = True
        #             break

        return HexState(is_final_state=final, player=player, board=board, k=n)

        # TODO? (performance gains): Make converter directly from "Flattened Game State" to ANET input without proxy through HexState

    @staticmethod
    def empty_board(starting_player: int = 1, k=cfg.k):
        board = np.zeros(k ** 2).reshape((k, k))
        return HexState(is_final_state=False, player=starting_player, board=board, k=k)

    def get_legal_actions(self) -> List[HexAction]:

        # TODO?: See if we can get faster than k^2

        actions = []
        if self.is_final_state:
            return actions
        for i in range(self.k):
            for j in range(self.k):
                if self.board[i][j] == 0:  # Empty space
                    actions.append(HexAction(row=i, col=j))

        return actions

    def get_all_actions(self) -> List[Action]:
        actions = []
        for i in range(self.k):
            for j in range(self.k):
                actions.append(HexAction(row=i, col=j))
        return actions

    def do_action(self, action: HexAction) -> State:

        board = self.board.copy()

        r = action.row
        c = action.col

        if board[r][c] != 0:
            raise Exception("Cannot place here :(")

        board[r][c] = self.player

        final = is_final_move(board=board, move=(r, c), player=self.player)

        return HexState(
            is_final_state=final, player=-self.player, board=board, k=self.k
        )

    def inverted(self):
        """
        Returns a new state where the board is transposed and the colors reversed
        """
        return HexState(
            is_final_state=self.is_final_state,
            player=-self.player,
            board=-self.board.T,
            k=self.k,
        )

    def rotate180(self):
        """
        Returns a new state where the board is rotated 180 degrees, as well as the inverted version of this state
        """

        return (
            HexState(
                is_final_state=self.is_final_state,
                player=self.player,
                board=np.rot90(self.board, 2),
                k=self.k,
            ),
            HexState(
                is_final_state=self.is_final_state,
                player=-self.player,
                board=-np.rot90(self.board, 2).T,
                k=self.k,
            ),
        )

    def as_vector(self, mode=2):
        """
        Returns the game state as a vector intended to use as input for the ANET.

        """
        if (
            mode == 0
        ):  # Basic, we want the network to see the game same way regardless of which player we are, so we transpose the board if the player is -1.
            vector = []
            if self.player == 1:
                board = self.board
            else:
                board = -self.board.T  # Transpose and swap colors
            for p in board.flatten():
                if p == 1:
                    vector.extend([1, 0])
                elif p == -1:
                    vector.extend([0, 1])
                else:
                    vector.extend([0, 0])
            return np.array(vector)

        if mode == 1:
            # kxkx4 array, 0: player 1 stones, 1: player 2 stones: 3: empty stones: 4: 1 if player 1,0 if player 2
            array = np.zeros((self.k, self.k, 4))

            array[:, :, 0] = self.board > 0
            array[:, :, 1] = self.board < 0
            array[:, :, 2] = self.board == 0
            array[:, :, 3] = self.player == 1
            return array

        if mode == 2:  # gao-2017 encoding

            array = np.zeros((self.k+4, self.k+4, 9))
            p = self.player
            b_copy = self.board.copy()

            b = np.pad(b_copy, 2, mode='constant', constant_values=(-1))

            b[0:2] = 1
            b[-2:] = 1

            array[:, :, 0] = b > 0
            array[:, :, 1] = b < 0
            array[:, :, 2] = b == 0
            array[:, :, 3] = p == 1
            array[:, :, 4] = p != 1

            # Bridges:

            player1_endpoint = np.zeros(b_copy.shape)
            player2_endpoint = np.zeros(b_copy.shape)
            save_bridge = np.zeros(b_copy.shape)
            form_bridge = np.zeros(b_copy.shape)

            for i in range(self.k):
                for j in range(self.k):
                    for pattern in bridges[(i, j)]:
                        carrier1 = pattern[0]
                        carrier2 = pattern[1]
                        endpoint = pattern[2]
                        # Bridge endpoint
                        if b_copy[carrier1] == 0 or b_copy[carrier2] == 0:

                            if b_copy[carrier1] != 1 and b_copy[carrier2] != 1:
                                if b_copy[i, j] == 1:
                                    # Player 1 bridge endpoint
                                    player1_endpoint[i, j] = max(
                                        player1_endpoint[i, j], b_copy[endpoint] >= 0)
                            elif b_copy[carrier1] != -1 and b_copy[carrier2] != -1:
                                if b_copy[i, j] == -1:
                                    # player 2 bridge endpoint
                                    player2_endpoint[i, j] = max(player2_endpoint[i, j],
                                                                 b_copy[endpoint] <= 0)
                        if b_copy[i, j] == p:
                            if b_copy[carrier1] == 0 and b_copy[carrier2] == 0:
                                # Form bridge
                                form_bridge[endpoint] = max(
                                    form_bridge[endpoint], b_copy[endpoint] == 0)

                            # Save bridge

                            if b_copy[carrier1] != 0 and b_copy[carrier2] == 0:
                                if b_copy[carrier1] != p:
                                    save_bridge[carrier2] = max(
                                        save_bridge[carrier2], b_copy[endpoint] == p)
                            if b_copy[carrier1] == 0 and b_copy[carrier2] != 0:
                                if b_copy[carrier2] != p:
                                    save_bridge[carrier1] = max(
                                        save_bridge[carrier1], b_copy[endpoint] == p)

            # player 1 bridge endpoints
            array[:, :, 5] = np.pad(player1_endpoint, 2)
            # player 2 bridge endpoints
            array[:, :, 6] = np.pad(player2_endpoint, 2)
            array[:, :, 7] = np.pad(form_bridge, 2)  # toplay form bridge
            array[:, :, 8] = np.pad(save_bridge, 2)  # toplay save bridge

            return array

    def to_array(self):
        arr = np.zeros(self.k ** 2 + 1)
        arr[0] = self.player
        arr[1:] = self.board.flatten()

        return arr

    @staticmethod
    def from_array_to_vector(array, mode=2):
        return HexState.from_array(array).as_vector(mode=mode)

    def plot(self, labels: bool = False):

        cdict = {0: "grey", 1: "red", -1: "blue"}

        G = nx.Graph()
        colormap = []
        for n in neighbors:
            G.add_node(n, label=str(n))
            colormap.append(cdict[self.board[n]])
        for key, value in neighbors.items():
            for n in value:
                G.add_edge(key, n)

        edgemap = []
        k = self.k
        for el in G.edges:
            key = el[0]
            n = el[1]
            if key[0] == 0 or key[0] == k - 1:
                if key[0] == n[0]:
                    edgemap.append("red")

                else:
                    edgemap.append("black")
            elif key[1] == 0 or key[1] == k - 1:
                if key[1] == n[1]:
                    edgemap.append("blue")
                else:
                    edgemap.append("black")
            else:
                edgemap.append("black")

        clear_output(wait=True)
        plt.figure(figsize=(10, 10))
        seed = 8
        if k == 5:
            seed = 10
        pos = nx.spring_layout(G, seed=seed)
        nx.draw(G, pos=pos, node_color=colormap,
                node_size=400, edge_color=edgemap)
        if labels:
            nx.draw_networkx_labels(G, pos, verticalalignment="center")
        plt.show()

    def copy(self):
        return copy.deepcopy(self)

    def __hash__(self):
        return hash(repr(self))


class HexWorld(GameWorld):
    # TODO: Scrap this? Task description wants a "State manager" though .__.

    def __init__(self):
        pass
