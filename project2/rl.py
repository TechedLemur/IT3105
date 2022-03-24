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


class ReinforcementLearningAgent:
    def __init__(self):
        self.anet = ActorNet(cfg.input_shape, cfg.output_length)

    def episode(self, ep: int, anet, rollout_chance=cfg.rollout_chance):
        print(f"Episode {ep}")

        # if ep % save_params_interval == 0:
        #     print("Saved network weights")
        #     self.anet.save_params(ep, suffix=file_suffix)

        world = HexState.empty_board()
        # player = -1
        mcts = MCTS(anet, world)
        move = 1
        x_train = []
        y_train = []

        reward_factors = np.array([])

        while not world.is_final_state:
            # print(f"Move {move}:")
            mcts.run_simulations(rollout_chance)

            # player = -player
            D = mcts.get_visit_counts_from_root()

            # print("D: ", D)

            x_train.append(mcts.root.state.as_vector())
            y_train.append(D)

            reward_factors *= cfg.reward_decay  # Decay rewards after each move
            reward_factors = np.append(reward_factors, 1)

            action = anet.select_action(world)

            world = world.do_action(action)
            mcts.root = mcts.root.children[action]
            i += 1
            move += 1

        winner = -world.player

        y_train_value = winner * reward_factors
        # the critic can be trained by using the score obtained at the end of each
        # actual game (i.e. episode) as the target value for backpropagation, wherein the net receives each state of the recent
        # episode as input and tries to map that state to the target (or a discounted version of the target)

        return x_train, y_train, y_train_value, winner

    def train(self, file_suffix=""):
        wins = np.zeros(cfg.episodes)

        x_size = list(cfg.input_shape)

        x_size.insert(0, cfg.replay_buffer_size)

        y_size = (cfg.replay_buffer_size, cfg.output_length)

        # These three constitutes the "replay buffer"
        x_train = np.zeros(x_size)
        y_train = np.zeros(y_size)
        y_train_value = np.zeros(cfg.replay_buffer_size)

        i = 0
        save_params_interval = cfg.episodes // cfg.M

        rollout_chance = cfg.rollout_chance

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
            buffer_reward_factors = np.array([])

            all_actions = world.get_all_actions()
            while not world.is_final_state:
                # print(f"Move {move}:")
                mcts.run_simulations(rollout_chance=rollout_chance)

                # player = -player
                D = mcts.get_visit_counts_from_root()

                # print("D: ", D)

                j = i % cfg.replay_buffer_size

                x_train[j, :] = mcts.root.state.as_vector()
                y_train[j, :] = D
                buffer_indices.append(j)

                buffer_reward_factors *= cfg.reward_decay  # Decay rewards after each move
                buffer_reward_factors = np.append(buffer_reward_factors, 1)

                # action = self.anet.select_action(world)
                # Choose actual move (a*) based on D

                # Round to avoid floating point error
                probs = np.around(D, 3)
                probs = probs / np.sum(probs)  # Normalize
                # Select action based on distribution D

                action = np.random.choice(all_actions, p=probs)

                world = world.do_action(action)
                # graph = mcts.draw_graph()
                # graph.render(f"mcts-graphs/graph{move}")
                mcts.root = mcts.root.children[action]
                i += 1
                move += 1

                # world.plot()

            rollout_chance *= cfg.rollout_decay  # Decay rollout chance

            winner = -world.player
            y_train_value[buffer_indices] = winner * buffer_reward_factors
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
