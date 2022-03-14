from actor_net import ActorNet
from mcts import MCTS
from gameworlds.nim_world import NimWorld
from config import Config as cfg
from gameworlds.gameworld import GameWorld


class ReinforcementLearningAgent:
    def __init__(self):
        self.mcts = MCTS()
        self.anet = ActorNet()
        pass

    def train(self, gameworld: GameWorld):
        for _ in range(cfg.episodes):
            gameworld.set_initial_state()

            self.mcts.run_simulations()

            self.anet.train()

