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
        self.anet = ActorNet(100, 100)
        pass

    def train(self):
        wins = np.zeros(cfg.episodes)

        replay_buffer = np.zeros((cfg.replay_buffer_size, 20))
        i = 0
        for ep in range(cfg.episodes):

            world = HexState.empty_board()
            player = -1
            mcts = MCTS(self.anet, world)

            while not world.is_final_state:
                mcts.run_simulations()

                player = -player
                D = mcts.get_visit_counts_from_root()
                D = D[::mcts.root.player]

                # replay_buffer[i % cfg.replay_buffer_size,:] = np.concatenate(mcts.root.state.as_vector(), D)

                # action = mcts.get_best_action()
                action = self.anet.select_action(world)
                world = world.do_action(action)
                i += 1

                # world.plot()

            wins[ep] = player
            # mini_batch = random.sample(replay_buffer, cfg.mini_batch_size)
            # print(mini_batch.size)
            # self.anet.train(mini_batch)

        print(f"Player 1 won {np.count_nonzero(wins)}% of the games!")
