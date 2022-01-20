from typing import List
from xmlrpc.client import Boolean
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

        self.discs = [[] for _ in range(self.nr_discs)]
        self.pegs = list(range(1, self.nr_pegs + 1))
        self.discs[0] = self.pegs

    def get_actions(self) -> List[HanoiWorldAction]:
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
        pass

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
