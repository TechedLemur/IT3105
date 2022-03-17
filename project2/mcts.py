from email.policy import default
from typing import List, Optional, Tuple
from config import Config as cfg

import numpy as np
from actor_net import ActorNet
from gameworlds.gameworld import GameWorld
from gameworlds.gameworld import Action, State
from collections import defaultdict


class MCTSNode:
    def __init__(self, parent: "MCTSNode", state: State):
        self.parent = parent
        self.state = state
        self.children: dict[Action, MCTSNode] = {}
        self.visits: int = 0

        if not self.parent:
            self.player = 1
        else:
            self.player = -self.parent.player

    def is_final_state(self) -> Tuple[bool, int]:
        """Check if it is a final state.

        Returns:
            Tuple[bool, int]: Returns both if this is a final game state and if so the game result (z_L). Either -1 (lose), 0 (draw) or +1 (win).
        """
        return (self.state.is_final_state, 1)


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, actor, world: GameWorld) -> None:
        self.root: MCTSNode = MCTSNode(None)
        self.Q: defaultdict(tuple(State, Action), int) = {}
        self.N: defaultdict(tuple(State, Action), int) = {}
        self.V_i = np.zeros(cfg.search_games)
        self.actor = actor
        self.world = world

    def update_with_new_root(self, state: State) -> None:
        self.root = MCTSNode(None, state)

    def run_simulations(self) -> None:
        self.d_s_a_i = defaultdict(lambda: np.zeros(cfg.search_games))
        self.visited: List[Tuple[State, Action]] = []
        for i in range(cfg.search_games):
            self.iteration = i
            self.apply_tree_policy()
            self.do_rollout()
            self.backpropagate()

    def backpropagate(self) -> None:
        """Backpropagate after a rollout from a leaf-node.
           Updates N(s,a) and Q(s,a).
        """

        for state, action in self.visited:
            self.N[(state, action)] = sum(self.d_s_a_i[(state, action)])
            self.Q[(state, action)] = (
                1
                / self.N[(state, action)]
                * sum(self.d_s_a_i[(state, action)] * self.V_i)
            )

    def do_rollout(self, start_node: MCTSNode) -> None:
        """Do a rollout from a leaf-node until a final state is found.
        The rollout is done by using the ActorNet to choose rollout actions repeatedly until
        a final state is found.

        Args:
            start_node (MCTSNode): Leaf-node to start rollout from.
        """

        game_finished = False
        current_node = start_node
        while not game_finished:
            action = self.actor.select_action(self.world)

            self.d_s_a_i[(current_node.state, action)][self.iteration] = 1
            self.visited.append((current_node.state, action))

            new_state = self.world.do_action(action)
            new_node = MCTSNode(current_node, new_state)
            current_node.children[action] = new_node
            current_node = new_node
            game_finished, z_L = new_node.is_final_state()
        self.V_i[self.iteration] = z_L

    def apply_tree_policy(self) -> None:
        """Apply the tree search policy from root until a leaf-node is found.

        We want to find the branch with the highest combination of Q(s, a) + u(s, a)
        We are using Upper Confidence Bound for Trees (UCT) for the exploration bonus u(s, a)
        """

        is_leaf_node = False
        current_node = self.root
        while not is_leaf_node:
            if not current_node.children:
                is_leaf_node = True
                break
            N_s = []
            N_s_a = []
            Q_s_a = []
            for a, c in current_node.children.items():
                N_s.append(c.visits)
                N_s_a.append(self.N_s_a((c.state, a)))
                Q_s_a.append(self.Q((c.state, a)))

            if current_node.player == 1:
                max_index = np.argmax(Q_s_a + MCTS.uct(N_s, N_s_a))
                best_action = current_node.children.keys[max_index]
            else:
                min_index = np.argmin(Q_s_a - MCTS.uct(N_s, N_s_a))
                best_action = current_node.children.keys[min_index]

            self.d_s_a_i[(current_node.state, best_action)][self.iteration] = 1
            self.visited.append((current_node.state, best_action))

            current_node = current_node.children[best_action]

    @staticmethod
    def uct(N_s, N_s_a) -> np.array:
        return cfg.c * np.sqrt(np.log2(N_s) / (1 + N_s_a))

if __name__ == "__main__":
    pass
