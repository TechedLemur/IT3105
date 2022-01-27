from dataclasses import dataclass
from typing import List
from simworlds.simworld import SimWorld, Action, State
from config import Config


@dataclass
class HanoiWorldAction(Action):
    dics_to_move: int
    to_peg: int

    def __hash__(self):
        return hash(repr(self))


@dataclass
class HanoiWorldState(State):
    state: List[List[int]]

    def __hash__(self):
        return hash(repr(self))


class HanoiWorld(SimWorld):
    def __init__(self):
        self.nr_pegs = Config.HanoiWorldConfig.PEGS
        self.nr_discs = Config.HanoiWorldConfig.DISCS

        self.pegs = HanoiWorldState([[] for _ in range(self.nr_discs)])
        self.pegs.state[0] = list(i * 2 + 1 for i in range(0, self.nr_discs))
        self.length = max(self.pegs.state[0]) + 2

    def __get_reward(self):

        return -1

    def get_legal_actions(self) -> List[HanoiWorldAction]:
        # Might be easier to just make check_legal_action public?
        pass

    def do_action(self, action: HanoiWorldAction) -> bool:
        if self.__check_legal_action(action):
            for peg in self.pegs.state:
                if action.disc_to_move in peg:
                    peg.remove(action.disc_to_move)
            self.pegss.state[action.to_peg].insert(0, action.disc_to_move)
            return True
        else:
            return False

    def get_state(self) -> HanoiWorldState:
        return self.pegs.state

    def visualize_individual_solutions(self):
        print()

        conc_str: List[str] = []

        for peg in self.pegs.state:
            tmp = ""
            ln = len(peg)
            while ln < self.nr_discs:
                tmp += " " * self.length + " \n"
                ln += 1
            for disc in peg:
                tmp += ("*" * disc).center(self.length, " ") + " \n"

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
