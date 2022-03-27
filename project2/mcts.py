import random
from typing import List, Optional, Tuple
from config import cfg
import uuid

import numpy as np
from actor_net import ActorNet
from gameworlds.gameworld import GameWorld
from gameworlds.gameworld import Action, State
from collections import defaultdict

from graphviz import Digraph

from gameworlds.nim_world import NimWorld
from gameworlds.hex_world import HexState


class MCTSNode:
    """MCTSNode-class which represents a node in the MCTS-tree. This also represents a search state.
    """

    def __init__(self, parent: "MCTSNode", state: State):
        self.parent = parent
        self.state = state
        self.children: dict[Action, MCTSNode] = {}
        self.visits = 0

        self.id = uuid.uuid4()

    def is_final_state(self) -> Tuple[bool, int]:
        """Check if it is a final state.

        Returns:
            Tuple[bool, int]: Returns both if this is a final game state and if so the game result (z_L).
            Either -1 (lose), 0 (draw) or +1 (win) depending on the type of game.
        """

        return (self.state.is_final_state, -self.state.player)

    def __repr__(self):
        return f"MCTSNode(state={self.state},visits={self.visits},player={self.state.player})"

    def __hash__(self):
        return self.id.int


class MCTS:
    """
    Monte Carlo Tree Search-class. This represents a tree which can run simulations.
    """

    def __init__(self, actor, state: State) -> None:
        self.root: MCTSNode = MCTSNode(None, state)
        self.Q: defaultdict[tuple[MCTSNode, Action], int] = defaultdict(int)
        self.N_s_a: defaultdict[tuple[MCTSNode,
                                      Action], int] = defaultdict(int)
        self.N_s: defaultdict[MCTSNode, int] = defaultdict(int)
        self.V_i = np.zeros(cfg.search_games)
        self.actor = actor

    def get_best_action(self) -> Action:
        """Get the best action to do as calculated by the arch from root with the most visits.

        Returns:
            Action: Action (arch) from root with most visits.
        """
        D = self.get_visit_counts_from_root()
        action = list(self.root.children.keys())[np.argmax(D)]
        self.root = self.root.children[action]
        return action

    def run_simulations(self, rollout_chance=cfg.rollout_chance) -> None:
        """Run M simulations from the current state.
        A simulation consists of:
        1. Applying tree policy to a leaf node.
        2. Doing a rollout from the leaf node.
        3. Backpropagating the eval which updates N(s, a) and Q(s, a).
        """

        self.d_s_a_i: defaultdict[tuple[MCTSNode, Action]] = defaultdict(
            lambda: np.zeros(cfg.search_games)
        )  # Represents the visits of arch on iteration i. (Using indice to get the ith visit count).
        for i in range(cfg.search_games):
            self.visited: List[
                Tuple[MCTSNode, Action]
            ] = []  # Visited nodes are appended here which is used during backpropagation
            self.current_world = self.root.state.copy()
            self.iteration = i
            leaf_node = self.apply_tree_policy()

            if random.random() < rollout_chance:
                self.do_rollout(leaf_node)
            else:  # Use critic
                is_finished, z = leaf_node.is_final_state()
                if not is_finished:
                    a, z = self.actor.get_action_and_reward(leaf_node.state)
                    # Add node to visited
                    self.visited.append((leaf_node, a))
                self.V_i[self.iteration] = z
            self.backpropagate()

    def backpropagate(self) -> None:
        """Backpropagate after a rollout from a leaf-node.
           Updates N(s,a) and Q(s,a).

           N(s,a) = sum_i d(s, a, i)
           Q(s, a) = 1/N(s, a) * sum_i d(s, a, i)*V(s^i)

           Where V(s^i) was the evaluation value of leaf node s on iteration i.
        """

        for node, action in self.visited:
            self.d_s_a_i[(node, action)][self.iteration] = 1
            self.N_s[node] += 1

            self.N_s_a[(node, action)] = sum(self.d_s_a_i[(node, action)])
            self.Q[(node, action)] = (
                1
                / self.N_s_a[(node, action)]
                * sum(self.d_s_a_i[(node, action)] * self.V_i)
            )

    def decay_rollout_chance(self):
        self.rollout_p *= cfg.rollout_decay

    def do_rollout(self, start_node: MCTSNode, add_rollout_nodes_to_tree=False) -> None:
        """Do a rollout from a leaf-node until a final state is found.
        The rollout is done by using the ActorNet to choose rollout actions repeatedly until
        a final state is found.

        Args:
            start_node (MCTSNode): Leaf-node to start rollout from.
            add_rollout_nodes_to_tree: (bool): If rollout nodes should be added to the tree.
        """

        player = start_node.state.player
        game_finished, z_L = start_node.is_final_state()
        it = 0
        current_node = start_node
        while not game_finished:
            if random.random() <= cfg.random_rollout_move_p:
                legal_actions = self.current_world.get_legal_actions()
                action = random.choice(legal_actions)
            else:
                action = self.actor.select_action(self.current_world)
            # The start-node is the leaf-node which should be regarded as visited.
            if it == 0:
                self.visited.append((start_node, action))
            new_state = self.current_world.do_action(action)

            if add_rollout_nodes_to_tree:
                self.visited.append((current_node, action))
                if action not in current_node.children.keys():
                    new_node = MCTSNode(current_node, new_state)
                    current_node.children[action] = new_node
                    current_node = new_node
                else:
                    current_node = current_node.children[action]

            game_finished = new_state.is_final_state

            player = -player
            it += 1

            self.current_world = new_state
        z_L = -player
        self.V_i[self.iteration] = z_L

    def tree_policy(self, node: MCTSNode, apply_exploraty_bonus=True) -> Action:
        """Get the tree policy from a node which is the action which maximizes both the Q-value and
        an exploratory bonus for going on branches not visited a lot. This is done by using the UCT-formula.


        Args:
            node (MCTSNode): Node to apply
            apply_exploraty_bonus (bool, optional): _description_. Defaults to True.

        Returns:
            Action: Best action according to the UCT-calculation.
        """

        N_s_a = np.array([self.N_s_a[(node, a)] for a in node.children.keys()])
        Q_s_a = np.array([self.Q[(node, a)] for a in node.children.keys()])

        if node.state.player == 1:
            action_values = Q_s_a + \
                MCTS.uct(self.N_s[node], N_s_a) * int(apply_exploraty_bonus)
            max_indices = np.flatnonzero(
                action_values == np.max(action_values))
        else:
            action_values = Q_s_a - \
                MCTS.uct(self.N_s[node], N_s_a) * int(apply_exploraty_bonus)
            max_indices = np.flatnonzero(
                action_values == np.min(action_values))

        # if not max_indices.any():
            # return random.choice(list(node.children.keys()))

        # Choose randomly between the best choices (if they have equal values)
        return list(node.children.keys())[np.random.choice(max_indices)]

    def apply_tree_policy(self) -> MCTSNode:
        """Apply the tree search policy from root until a leaf-node is found.

        We want to find the branch with the highest combination of Q(s, a) + u(s, a).
        We are using Upper Confidence Bound for Trees (UCT) for the exploration bonus u(s, a).

        Returns:
            MCTSNode: Leaf-node found by applying the tree-policy down the tree.
        """

        is_leaf_node = False
        current_node = self.root
        while not is_leaf_node:
            if not current_node.children:
                is_leaf_node = True
                break
            best_action = self.tree_policy(current_node)
            self.current_world = self.current_world.do_action(best_action)
            self.visited.append((current_node, best_action))
            current_node = current_node.children[best_action]
        # Expand leaf-node.
        for action in self.current_world.get_legal_actions():
            current_node.children[action] = MCTSNode(
                current_node, self.current_world.do_action(action)
            )
        return current_node

    @staticmethod
    def uct(N_s: int, N_s_a: np.array) -> np.array:
        """The UCT-calculation.

        Args:
            N_s (int): Number of visits for node.
            N_s_a (np.array): Number of visits of arches from node.

        Returns:
            np.array: UCT-calculation.
        """
        return cfg.c * np.sqrt(np.log(N_s) / (1 + N_s_a))

    def get_visit_counts_from_root(self) -> np.array:
        """Get the visit count distribution from root.

        Returns:
            np.array: Visit count distribution from root.
        """
        visit_counts = np.zeros((cfg.k, cfg.k))
        for a in self.root.children.keys():
            visit_counts[a.row][a.col] = self.N_s_a[self.root, a]
        # Make a distribution
        return (visit_counts / np.sum(visit_counts)).flatten()

    def draw_graph(self) -> Digraph:
        """Generate a digraph of the current tree which can be used
        for visualization purposes.

        Returns:
            Digraph: A digraph representation of the current tree.
        """

        dot = Digraph(format="png")

        nodes_to_visit = [self.root]
        node = nodes_to_visit[0]
        dot.node(
            str(node.state) + str(node.id), f"N: {self.N_s[node]}",
        )
        while nodes_to_visit:
            node = nodes_to_visit.pop()

            for key, val in node.children.items():
                dot.node(
                    name=str(val.state) + str(val.id), label=f"N: {self.N_s[val]}",
                )
                dot.edge(
                    str(node.state) + str(node.id),
                    str(val.state) + str(val.id),
                    f"Move: ({key.row}, {key.col}), Q: {self.Q[(node, key)]:.2f}, N: {self.N_s_a[(node, key)]}",
                )
                nodes_to_visit.append(val)
        return dot


if __name__ == "__main__":
    anet = ActorNet(input_shape=10, output_dim=10)
    wins = np.zeros(100)

    for i in range(100):
        world = HexState.empty_board()
        tree = MCTS(anet, world)
        player = -1
        j = 0
        while not world.is_final_state:
            tree.run_simulations()
            player = -player
            D = tree.get_visit_counts_from_root()
            graph = tree.draw_graph()
            graph.render(f"mcts-graphs/graph{j}")
            j += 1
            action = tree.get_best_action()
            world = world.do_action(action)
        print(f"Player {player} won!")
        wins[i] = player

        if player != 1:
            break
    print(f"Player 1 won {np.average(wins)*100}%")
