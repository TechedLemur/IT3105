from typing import List
from simworlds.simworld import SimWorld, Action
from config import Config


class HanoiWorldAction(Action):
    def __init__(self, peg_to_move: int, to_disc: int):
        self.peg = peg_to_move
        self.disc = to_disc


class HanoiWorld(SimWorld):
    def __init__(self):
        self.nr_pegs = Config.HanoiWorldConfig.PEGS
        self.nr_discs = Config.HanoiWorldConfig.DISCS

        self.discs: List[List[int]] = [[] for _ in range(self.nr_discs)]
        self.discs[0] = list(i * 2 + 1 for i in range(0, self.nr_pegs))
        self.length = max(self.discs[0]) + 2

    def get_actions(self) -> List[HanoiWorldAction]:
        # Might be easier to just make check_legal_action public?
        pass

    def do_action(self, action: HanoiWorldAction) -> bool:
        if self.__check_legal_action(action):
            for disc in self.discs:
                if action.peg in disc:
                    disc.remove(action.peg)
            self.discs[action.disc].insert(0, action.peg)
            return True
        else:
            return False

    def get_state(self):
        pass

    def visualize_individual_solutions(self):
        print()

        conc_str: List[str] = []

        for disc in self.discs:
            tmp = ""
            ln = len(disc)
            while ln < self.nr_pegs:
                tmp += " " * self.length + " \n"
                ln += 1
            for peg in disc:
                tmp += ("*" * peg).center(self.length, " ") + " \n"

            tmp += "_" * self.length + " \n"
            conc_str.append(tmp)

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
        for disc in self.discs:
            if action.peg in disc:
                if (
                    disc[0] != action.peg
                ):  # The peg we are trying to move should be the smallest in the disc (the first in the list)
                    return False

        # If there already is a peg at the disc to move to check if the peg we are trying
        # to move is smaller than the peg there
        if len(self.discs[action.disc]) > 0 and self.discs[action.disc][0] < action.peg:
            return False

        return True
