import numpy as np


class Player:

    def __init__(self, anet, mcts) -> None:
        self.anet = anet
        self.mcts = mcts

    def get_action(self, state, timeout=600):

        legal = state.get_legal_actions()
        self.mcts.root = self.mcts.get_node_from_state(state)

        p, forced = self.anet.get_policy(state)

        self.mcts.run_simulations(time_out=timeout)
        if forced:
            a = legal[np.argmax(p)]
            self.mcts.root = self.mcts.root.children[a]

            return a

        D = self.mcts.get_D()

        c = 0.5
        P = c*p + (1-c)*D

        a = legal[np.argmax(P)]
        self.mcts.root = self.mcts.root.children[a]
        return a
