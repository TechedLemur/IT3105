import random
from re import I
from actor_net import ActorNet
from mcts import MCTS
from gameworlds.nim_world import NimWorld
from gameworlds.hex_world import HexState
from config import Config as cfg
from gameworlds.gameworld import GameWorld
import numpy as np
from collections import deque
import timeit
from multiprocessing import Pool, cpu_count


class ReinforcementLearningAgent:
    def __init__(self):
        self.anet = ActorNet(cfg.input_shape, cfg.output_length)

    @staticmethod
    def episode(params):

        anet_weigths = params[0]
        starting_player = params[1]
        rollout_chance = params[2]
        # if ep % save_params_interval == 0:
        #     print("Saved network weights")
        #     self.anet.save_params(ep, suffix=file_suffix)

        world = HexState.empty_board(starting_player=starting_player)
        # player = -1
        anet = ActorNet(cfg.input_shape, cfg.output_length)
        anet.set_weights(anet_weigths)
        mcts = MCTS(anet, world)
        move = 1
        x_train = []
        y_train = []

        reward_factors = []
        all_actions = world.get_all_actions()
        while not world.is_final_state:
            # print(f"Move {move}:")
            mcts.run_simulations(rollout_chance)

            # player = -player
            D = mcts.get_visit_counts_from_root()

            # print("D: ", D)

            x_train.append(mcts.root.state.as_vector())
            y_train.append(D)

            # Decay rewards after each move
            reward_factors = [x*cfg.reward_decay for x in reward_factors]
            reward_factors.append(1)

            # Choose actual move (a*) based on D
            # Round to avoid floating point error
            probs = np.around(D, 3)
            probs = probs / np.sum(probs)  # Normalize
            # Select action based on distribution D

            action = np.random.choice(all_actions, p=probs)

            world = world.do_action(action)
            mcts.root = mcts.root.children[action]
            move += 1

        winner = -world.player

        y_train_value = winner * np.array(reward_factors)
        # the critic can be trained by using the score obtained at the end of each
        # actual game (i.e. episode) as the target value for backpropagation, wherein the net receives each state of the recent
        # episode as input and tries to map that state to the target (or a discounted version of the target)
        return np.array(x_train), np.array(y_train), y_train_value, winner

    def train(self, file_suffix="", n_parallel=8):
        wins = []

        x_train = np.array([])
        y_train = np.array([])
        y_train_value = np.array([])

        i = 0
        save_params_interval = cfg.episodes // cfg.M

        rollout_chance = cfg.rollout_chance

        starting_player = 1

        for ep in range(cfg.episodes):
            print(f"Episode {ep}")

            if ep % save_params_interval == 0:
                print("Saved network weights")
                self.anet.save_params(ep, suffix=file_suffix)

            n = min(n_parallel, cpu_count())
            print(f"Running {n} games in parallel")

            weights = self.anet.get_weights()

            params = [(weights, starting_player, rollout_chance)
                      for _ in range(n)]

            with Pool(processes=n) as pool:

                result = pool.map(self.episode, params)

            for r in result:
                if not x_train.size:
                    x_train = r[0]
                    y_train = r[1]
                    y_train_value = r[2]
                else:
                    x_train = np.concatenate((x_train, r[0]))
                    y_train = np.concatenate((y_train, r[1]))
                    y_train_value = np.concatenate((y_train_value, r[2]))
                wins.append(r[3])

            # mini_batch = np.random.choice(
            #     min(cfg.replay_buffer_size, i),
            #     min(cfg.mini_batch_size, i),
            #     replace=False,
            # )
            # print()
            # print(x_train[mini_batch])
            # print(y_train[mini_batch])
            self.anet.train(x_train, y_train,
                            y_train_value=y_train_value)
            # self.anet.train(x_train[mini_batch], y_train[mini_batch],
            #                 y_train_value=y_train_value[mini_batch])
        wins = np.array(wins)
        self.anet.save_params(ep, suffix=file_suffix)
        print(
            f"Player 1 won {100*np.count_nonzero(wins[wins == 1])/wins.shape[0]:.2f}% of the games!"
        )
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_value = y_train_value
