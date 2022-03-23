import random
from actor_net import ActorNet
from mcts import MCTS
from gameworlds.nim_world import NimWorld
from gameworlds.hex_world import HexState
from config import Config as cfg
from gameworlds.gameworld import GameWorld
import numpy as np
from collections import deque


class ReinforcementLearningAgent:
    def __init__(self):
        self.anet = ActorNet(2 * cfg.k ** 2, cfg.k ** 2)

    def train(self, file_suffix=""):
        wins = np.zeros(cfg.episodes)

        # These two constitutes the "replay buffer"
        x_train = np.zeros((cfg.replay_buffer_size, 2 * cfg.k ** 2))
        y_train = np.zeros((cfg.replay_buffer_size, cfg.k ** 2))
        y_train_value = np.zeros((cfg.replay_buffer_size, 1))

        i = 0
        save_params_interval = cfg.episodes // cfg.M

        for ep in range(cfg.episodes):
            print(f"Episode {ep}")

            if ep % save_params_interval == 0:
                print("Saved network weights")
                self.anet.save_params(ep, suffix=file_suffix)

            world = HexState.empty_board()
            # player = -1
            mcts = MCTS(self.anet, world)
            move = 1

            buffer_indices = []
            while not world.is_final_state:
                # print(f"Move {move}:")
                mcts.run_simulations()

                # player = -player
                D = mcts.get_visit_counts_from_root()

                # print("D: ", D)

                x_train[i % cfg.replay_buffer_size,
                        :] = mcts.root.state.as_vector()
                y_train[i % cfg.replay_buffer_size, :] = D
                buffer_indices.append(i)

                action = self.anet.select_action(world)

                world = world.do_action(action)
                # graph = mcts.draw_graph()
                # graph.render(f"mcts-graphs/graph{move}")
                mcts.root = mcts.root.children[action]
                i += 1
                move += 1

                # world.plot()
            winner = -world.player
            y_train_value[buffer_indices] = winner
            # the critic can be trained by using the score obtained at the end of each
            # actual game (i.e. episode) as the target value for backpropagation, wherein the net receives each state of the recent
            # episode as input and tries to map that state to the target (or a discounted version of the target)

            wins[ep] = winner
            mini_batch = np.random.choice(
                min(cfg.replay_buffer_size, i),
                min(cfg.mini_batch_size, i),
                replace=False,
            )
            # print()
            # print(x_train[mini_batch])
            # print(y_train[mini_batch])
            self.anet.train(x_train[mini_batch], y_train[mini_batch],
                            y_train_value=y_train_value[mini_batch])
        self.anet.save_params(ep, suffix=file_suffix)
        print(
            f"Player 1 won {100*np.count_nonzero(wins[wins == 1])/wins.shape[0]:.2f}% of the games!"
        )
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_value = y_train_value
