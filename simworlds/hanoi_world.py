from dataclasses import dataclass
from typing import List, Tuple
from simworlds.simworld import SimWorld, Action, State
from config import Config
from copy import deepcopy
import numpy as np
import itertools
import string

# Helper for one-hot encoding


def generate_dictionary():
    """Generates a dictionary with string representation of a state as key and the index it should map to as value.
    See: https://en.wikipedia.org/wiki/Tower_of_Hanoi#/media/File:Tower_of_hanoi_graph.svg
    """
    pegs = Config.HanoiWorldConfig.PEGS
    s = list(string.ascii_lowercase[:pegs])

    product = [p for p in itertools.product(s, repeat=Config.HanoiWorldConfig.DISCS)]
    d = dict(zip(product, range(len(product))))
    return d


@dataclass
class HanoiWorldAction(Action):
    disc_to_move: int
    to_peg: int

    def __hash__(self):
        return hash(repr(self))


@dataclass
class HanoiWorldState(State):
    state: List[List[int]]

    def as_vector(self) -> np.ndarray:
        one_hot_state = np.zeros(
            Config.HanoiWorldConfig.ONE_HOT_LENGTH, dtype=np.float32
        )
        s = self.get_string_representation()
        one_hot_state[HanoiWorld.ONE_HOT_MAPPING[s]] = 1

        return one_hot_state

    def get_string_representation(self) -> str:
        """Generate a string representation for the current state.
        See here: https://en.wikipedia.org/wiki/Tower_of_Hanoi#/media/File:Tower_of_hanoi_graph.svg
        For example on the form "aaa" for the initial state in a 3x3 world.

        Returns:
            str: String representation of its state.
        """
        s = ["" for _ in range(Config.HanoiWorldConfig.DISCS)]

        for i in range(len(self.state)):
            for l in self.state[i]:
                s[l - 1] = chr(i + 97)
        return tuple(s)

    def __hash__(self):
        return hash(repr(self))


class HanoiWorld(SimWorld):

    ONE_HOT_MAPPING = generate_dictionary()

    def __init__(self):
        self.nr_pegs = Config.HanoiWorldConfig.PEGS
        self.nr_discs = Config.HanoiWorldConfig.DISCS

        self.set_initial_world_state()

    def set_initial_world_state(self):
        """Sets the world to its initial state.
        """
        self.pegs: HanoiWorldState = HanoiWorldState(
            False, [[] for _ in range(self.nr_pegs)]
        )
        self.pegs.state[0] = list(range(self.nr_discs))
        self.moves = 0

        self.state_history = [deepcopy(self.pegs.state)]

    def __get_reward(self) -> int:
        """Get the reward for the current state.

        Returns:
            int: _description_
        """
        if self.__is_final_state():  # Give a lot of reward if it managed to solve it.
            return 15
        return -1  # Punish it for every move. (Should hopefully minimize moves.)

    def __is_final_state(self) -> bool:
        """Check if final state.

        Returns:
            bool: If final state.
        """
        for i, p in enumerate(self.pegs.state):
            if (
                i != 0 and len(p) == self.nr_discs
            ):  # If a peg has the length of the maximum number of discs and it is not the first peg then it has solved this game!
                return True
        return False

    def get_legal_actions(self) -> List[HanoiWorldAction]:
        """Get all legal actions in the current state.

        Returns:
            List[HanoiWorldAction]: Legal actions in current state.
        """
        possible_actions: List[HanoiWorldAction] = []
        # For each peg we will iterate over each disc in the peg and see if we can move it.
        for peg in self.pegs.state:
            if peg:
                disc = peg[0]
                for i, p in enumerate(self.pegs.state):
                    if (
                        not p or disc < p[0]
                    ):  # Can move it if the peg is empty or the disc is smaller than the top disc in the peg.
                        possible_actions.append(HanoiWorldAction(disc, i))
        return possible_actions

    def do_action(self, action: HanoiWorldAction) -> Tuple[HanoiWorldState, int]:
        """Do an action.

        Args:
            action (HanoiWorldAction): Action to perform.

        Returns:
            Tuple[HanoiWorldState, int]: New state after action and its reward.
        """
        for peg in self.pegs.state:
            if action.disc_to_move in peg:
                peg.remove(action.disc_to_move)  # Remove disc from its current peg.
        # Insert the disc to its new peg at the top.
        self.pegs.state[action.to_peg].insert(0, action.disc_to_move)

        reward = self.__get_reward()
        final_state = self.__is_final_state()
        self.moves += 1
        new_state = HanoiWorldState(final_state, deepcopy(self.pegs.state))
        self.state_history.append(deepcopy(self.pegs.state))
        return (new_state, reward)

    def get_state(self) -> HanoiWorldState:
        """Get the current state.

        Returns:
            HanoiWorldState: The current state.
        """
        return HanoiWorldState(self.__is_final_state(), deepcopy(self.pegs.state))

    @staticmethod
    def visualize_solution(peg_states: List[List[List[int]]]):
        """Visualizes all the moves in a list of states.

        Args:
            peg_states (List[List[List[int]]]): States to visualize.
        """
        discs = Config.HanoiWorldConfig.DISCS
        for i, pegs in enumerate(peg_states):
            if i == 0:
                print("Initial state: ")
            else:
                print(f"Step {i}:")

            conc_str: List[str] = []
            length = discs + 7

            # Basically just generate the string to be shown.
            for peg in pegs:
                tmp = ""
                ln = len(peg)
                while ln < discs:
                    tmp += " " * length + " \n"  # Empty string for formatting reasons.
                    ln += 1
                for disc in peg:
                    tmp += ("*" * (2 * disc + 1)).center(
                        length, " "
                    ) + " \n"  # This string will represent a disc.

                tmp += "_" * length + " \n"  # Represents all the pegs.
                conc_str.append(tmp)

            # Copied from: https://stackoverflow.com/questions/60876606/how-to-concatenate-two-formatted-strings-in-python
            splt_lines = zip(
                conc_str[0].split("\n"), conc_str[1].split("\n")
            )  # Concatenate all pegs horizontally.
            # horizontal join
            final = "\n".join([x + y for x, y in splt_lines])

            for i in range(2, len(conc_str)):
                splt_lines = zip(
                    final.split("\n"), conc_str[i].split("\n")
                )  # Concatenate all pegs horizontally.
                # horizontal join
                final = "\n".join([x + y for x, y in splt_lines])
            print(final)
            print()

