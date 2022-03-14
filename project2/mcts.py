from typing import List, Optional
from config import Config as cfg

import numpy as np
from actor_net import ActorNet
from gameworlds.gameworld import GameWorld
from gameworlds.gameworld import Action, State


class MCTSNode:
    def __init__(self, parent: "MCTSNode", state: State):
        self.parent = parent
        self.state = state
        self.children: dict[Action, MCTSNode] = {}
        self.visits: int = 0
        self.player = 0

    def is_final_state(self) -> bool:
        pass


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, actor, world: GameWorld) -> None:
        self.root: MCTSNode = MCTSNode(None)
        self.Q: dict(tuple(State, Action), int) = {}
        self.N_s_a: dict(tuple(State, Action), int) = {}
        self.actor = actor
        self.world = world

    def update_with_new_root(self, state: State) -> None:
        self.root = MCTSNode(None, state)

    def run_simulations(self) -> None:
        for _ in range(cfg.search_games):
            self.apply_tree_policy()
            self.do_rollout()
            self.backpropagate()

    def backpropagate(self) -> None:
        pass

    def do_rollout(self, start_node: MCTSNode) -> List[Action]:
        """Do a rollout from a leaf-node until a final state is found.
        The rollout is done by using the ActorNet to choose rollout actions repeatedly until
        a final state is found.

        Args:
            start_node (MCTSNode): Leaf-node to start rollout from.

        Returns:
            List[Action]: Rollout actions.
        """

        game_finished = False
        path: List[Action] = []
        current_node = start_node
        while not game_finished:
            action = self.actor.select_action(self.world)
            path.append(action)
            new_state = self.world.do_action(action)
            new_node = MCTSNode(current_node, new_state)
            current_node.children[action] = new_node
            current_node = new_node
            game_finished = new_node.is_final_state()
        return path

    def apply_tree_policy(self) -> MCTSNode:
        """Apply the tree search policy from root until a leaf-node is found.

        We want to find the branch with the highest combination of Q(s, a) + u(s, a)
        We are using Upper Confidence Bound for Trees (UCT) for the exploration bonus u(s, a)

        Returns:
            MCTSNode: Leaf node found applying tree policy
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
            # Todo: Handle 2 players?
            max_index = np.argmax(Q_s_a + MCTS.uct(N_s, N_s_a))
            best_action = current_node.children.keys[max_index]
            current_node = current_node.children[best_action]
        return current_node

    @staticmethod
    def uct(N_s, N_s_a) -> np.array:
        return cfg.c * np.sqrt(np.log2(N_s) / (1 + N_s_a))


if __name__ == "__main__":
    mctsnode = MCTSNode(None, None)
