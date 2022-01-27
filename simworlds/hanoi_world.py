from dataclasses import dataclass
from typing import List, Tuple
from simworlds.simworld import SimWorld, Action, State
from config import Config
from copy import deepcopy


@dataclass
class HanoiWorldAction(Action):
    disc_to_move: int
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

        self.pegs: HanoiWorldState = HanoiWorldState(
            False, [[] for _ in range(self.nr_discs)]
        )
        self.pegs.state[0] = list(i * 2 + 1 for i in range(0, self.nr_discs))
        self.length = max(self.pegs.state[0]) + 2

    def __get_reward(self) -> int:
        return -1

    def __is_final_state(self):
        for i, p in self.pegs.state:
            if i != 0:
                if len(p) == self.nr_discs:
                    return True
        return False

    def get_legal_actions(self) -> List[HanoiWorldAction]:
        possible_actions: List[HanoiWorldAction] = []
        for peg in self.pegs.state:
            if peg:
                disc = peg[0]
                for i, p in enumerate(self.pegs.state):
                    if not p or disc < p[0]:
                        possible_actions.append(HanoiWorldAction(disc, i))
        return possible_actions

    def do_action(self, action: HanoiWorldAction) -> Tuple[HanoiWorldState, int]:
        for peg in self.pegs.state:
            if action.disc_to_move in peg:
                peg.remove(action.disc_to_move)
        self.pegs.state[action.to_peg].insert(0, action.disc_to_move)
        reward = self.__get_reward()
        final_state = self.__is_final_state()
        return (HanoiWorldState(final_state, deepcopy(self.pegs.state)), reward)

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
