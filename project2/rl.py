from actor_net import ActorNet
from mcts import MCTS
from gameworlds.nim_world import NimWorld
from config import Config as cfg
from gameworlds.gameworld import GameWorld
import numpy as np


class ReinforcementLearningAgent:
    def __init__(self):
        self.anet = ActorNet()
        pass

    def train(self):
        wins = np.zeros(cfg.episodes)

        for ep in range(cfg.episodes):
            world = NimWorld(K=2, N=5, current_pieces=5)
            player = -1
            mcts = MCTS(self.anet, world)

            replay_buffer = []
            while not world.get_state().is_final_state:
                mcts.run_simulations()

                player = -player
                D = mcts.get_visit_counts_from_root()
                replay_buffer.append((D, mcts.root))

                action = mcts.get_best_action()
                world.do_action(action)
            wins[ep] = player
            # self.anet.train(replay_buffer)

        print(f"Player 1 won {np.count_nonzero(wins)}% of the games!")

