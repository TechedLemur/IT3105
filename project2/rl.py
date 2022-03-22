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
        print(cfg.k)
        self.anet = ActorNet(2*cfg.k**2, cfg.k**2)

    def train(self):
        wins = np.zeros(cfg.episodes)

        # These two constitutes the "replay buffer"
        x_train = np.zeros((cfg.replay_buffer_size,2*cfg.k**2))
        y_train = np.zeros((cfg.replay_buffer_size,cfg.k**2))

        i = 0
        for ep in range(cfg.episodes):

            world = HexState.empty_board()
            player = -1
            mcts = MCTS(self.anet, world)

            while not world.is_final_state:
                mcts.run_simulations()

                player = -player
                D = mcts.get_visit_counts_from_root()

                x_train[i % cfg.replay_buffer_size, :] = mcts.root.state.as_vector()
                y_train[i % cfg.replay_buffer_size, :] = D[::mcts.root.state.player]

                # action = mcts.get_best_action()
                action = self.anet.select_action(world)
                world = world.do_action(action)
                i += 1

                # world.plot()

            wins[ep] = player
            mini_batch = np.random.choice(
                cfg.replay_buffer_size, cfg.mini_batch_size, replace=False
            )
            self.anet.train(x_train[mini_batch], y_train[mini_batch])

        print(
            f"Player 1 won {np.count_nonzero(wins[wins == 1])/wins.shape[0]:.2f}% of the games!"
        )
