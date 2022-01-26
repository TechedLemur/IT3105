from dataclasses import dataclass
from typing import List
from simworlds.simworld import SimWorld, Action, State
from config import Config


@dataclass
class HanoiWorldAction(Action):
    peg_to_move: int
    to_disc: int


@dataclass
class HanoiWorldState(State):
    state: List[List[int]]


class HanoiWorld(SimWorld):
    def __init__(self):
        self.nr_pegs = Config.HanoiWorldConfig.PEGS
        self.nr_discs = Config.HanoiWorldConfig.DISCS

        self.discs = HanoiWorldState([[] for _ in range(self.nr_discs)])
        self.discs.state[0] = list(i * 2 + 1 for i in range(0, self.nr_pegs))
        self.length = max(self.discs.state[0]) + 2

    def get_legal_actions(self) -> List[HanoiWorldAction]:
        # Might be easier to just make check_legal_action public?
        pass

    def do_action(self, action: HanoiWorldAction) -> bool:
        if self.__check_legal_action(action):
            for disc in self.discs.state:
                if action.peg_to_move in disc:
                    disc.remove(action.peg_to_move)
            self.discs.state[action.to_disc].insert(0, action.peg_to_move)
            return True
        else:
            return False

    def get_state(self) -> HanoiWorldState:
        return self.discs.state

    def visualize_individual_solutions(self):
        print()

        conc_str: List[str] = []

        for disc in self.discs.state:
            tmp = ""
            ln = len(disc)
            while ln < self.nr_pegs:
                tmp += " " * self.length + " \n"
                ln += 1
            for peg in disc:
                tmp += ("*" * peg).center(self.length, " ") + " \n"

            tmp += "_" * self.length + " \n"
            conc_str.append(tmp)

        # Copied from: https://stackoverflow.com/questions/60876606/how-to-concatenate-two-formatted-strings-in-python
        splt_lines = zip(conc_str[0].split("\n"), conc_str[1].split("\n"))
        # horizontal join
        final = "\n".join([x + y for x, y in splt_lines])

        for i in range(2, len(conc_str)):
            splt_lines = zip(final.split("\n"), conc_str[i].split("\n"))
            # horizontal join
            final = "\n".join([x + y for x, y in splt_lines])
        print(final)
        print()

    def visualize_learning_progress(self):
        pass

    def __check_legal_action(self, action: HanoiWorldAction):
        for disc in self.discs.state:
            if action.peg_to_move in disc:
                if (
                    disc[0] != action.peg_to_move
                ):  # The peg we are trying to move should be the smallest in the disc (the first in the list)
                    return False

        # If there already is a peg at the disc to move to check if the peg we are trying
        # to move is smaller than the peg there
        if (
            len(self.discs.state[action.to_disc]) > 0
            and self.discs.state[action.to_disc][0] < action.peg_to_move
        ):
            return False

        return True
