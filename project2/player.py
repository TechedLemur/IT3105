import numpy as np


class Player:

    def __init__(self, anet, mcts) -> None:
        self.anet = anet
        self.mcts = mcts
        self.failed = False

    def get_action(self, state, timeout=600):

        legal = state.get_legal_actions()
        new_root = None
        if not self.failed:
            new_root = self.mcts.get_node_from_state(state)
        if new_root:
            self.mcts.root = new_root

        else:
            self.failed = True

        p, forced = self.anet.get_policy(state)
        if not self.failed:
            self.mcts.run_simulations(time_out=timeout)
        if forced:
            a = legal[np.argmax(p)]
            if not self.failed:
                self.mcts.root = self.mcts.root.children[a]

            return a
        if not self.failed:
            D = self.mcts.get_D()

        c = 0.6
        if not self.failed:
            P = c*p + (1-c)*D
        else:
            P = p

        a = legal[np.argmax(P)]
        if not self.failed:
            self.mcts.root = self.mcts.root.children[a]
        return a
