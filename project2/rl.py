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
from datetime import datetime


class ReinforcementLearningAgent:
    def __init__(self):
        self.anet = ActorNet(cfg.input_shape, cfg.output_length)

    @staticmethod
    def episode(params, anet=None):
        t1 = timeit.default_timer()
        anet_weigths = params[0]
        starting_player = params[1]
        rollout_chance = params[2]
        epsilon = params[3]

        world = HexState.empty_board(starting_player=starting_player)
        if not anet:
            anet = ActorNet(cfg.input_shape, cfg.output_length)
            anet.set_weights(anet_weigths)
            anet.epsilon = epsilon

        mcts = MCTS(anet, world)
        move = 1
        x_train = []
        y_train = []
        states = []
        reward_factors = []
        all_actions = world.get_all_actions()
        while not world.is_final_state:
            # print(f"Move {move}:")
            mcts.run_simulations(rollout_chance)

            # player = -player
            D = mcts.get_visit_counts_from_root()

            # print("D: ", D)
            # Decay rewards after each move
            reward_factors = [x*cfg.reward_decay for x in reward_factors]
            state = mcts.root.state
            D_matrix = D.reshape((state.k, state.k))

            states.append(state.to_array())
            # Add training cases, and their symmetric versions
            x_train.append(state.as_vector())
            y_train.append(D)
            reward_factors.append(1)

            inverted = state.inverted()
            x_train.append(inverted.as_vector())
            y_train.append(D_matrix.T.flatten())
            reward_factors.append(-1)
            states.append(inverted.to_array())

            rot, invRot = state.rotate180()

            x_train.append(rot.as_vector())
            y_train.append(np.rot90(D_matrix, 2).flatten())
            reward_factors.append(1)
            states.append(rot.to_array())

            x_train.append(invRot.as_vector())
            y_train.append(np.rot90(D_matrix.T, 2).flatten())
            reward_factors.append(-1)
            states.append(invRot.to_array())

            # Choose actual move (a*) based on D
            # Round to avoid floating point error
            probs = np.around(D, 3)
            probs = probs / np.sum(probs)  # Normalize
            # Select action based on distribution D

            action = np.random.choice(all_actions, p=probs)

            world = world.do_action(action)
            mcts.root = mcts.root.children[action]
            move += 1
        t2 = timeit.default_timer()
        winner = -world.player

        print(f"Game finished - used {t2-t1} seconds")
        y_train_value = winner * np.array(reward_factors)
        # the critic can be trained by using the score obtained at the end of each
        # actual game (i.e. episode) as the target value for backpropagation, wherein the net receives each state of the recent
        # episode as input and tries to map that state to the target (or a discounted version of the target)
        return np.array(x_train), np.array(y_train), y_train_value, winner, np.array(states)

    def train(self, file_suffix="", n_parallel=8):
        wins = []

        x_train = np.array([])
        y_train = np.array([])
        y_train_value = np.array([])
        states = np.array([])
        i = 0
        save_params_interval = cfg.episodes // cfg.M

        rollout_chance = cfg.rollout_chance

        starting_player = 1

        n = min(n_parallel, cpu_count())

        for ep in range(cfg.episodes):
            print(f"Episode {ep}")

            if ep % save_params_interval == 0 and ep > 0:
                print("Saved network weights")
                self.anet.save_params(ep, suffix=file_suffix)

                timestamp = datetime.now().isoformat()[:19]
                # Save data for possible later training
                timestamp = timestamp.replace(":", "-")
                with open(f'project2/data/{timestamp}_ep_{ep}.npy', 'wb') as f:
                    np.save(f, states)
                    np.save(f, y_train)
                    np.save(f, y_train_value)
                print("Saved data")

            if n > 1:  # Parallel simulations
                print(f"Running {n} games in parallel")

                weights = self.anet.get_weights()
                epsilon = self.anet.epsilon

                params = [(weights, starting_player, rollout_chance, epsilon)
                          for _ in range(n)]

                with Pool(processes=n) as pool:

                    result = pool.map(self.episode, params)
            else:
                print("Running on single thread")
                params = (None, starting_player, rollout_chance, None)
                result = [self.episode(params, self.anet)]

            for r in result:

                if not x_train.size:
                    x_train = r[0]
                    y_train = r[1]
                    y_train_value = r[2]
                    states = r[4]
                else:
                    x_train = np.concatenate((x_train, r[0]))
                    y_train = np.concatenate((y_train, r[1]))
                    y_train_value = np.concatenate((y_train_value, r[2]))
                    states = np.concatenate((states, r[4]))
                wins.append(starting_player == r[3])

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

            self.anet.update_epsilon(n=n)
            rollout_chance *= cfg.rollout_decay ** n
        wins = np.array(wins).astype(np.float32)
        self.anet.save_params(ep, suffix=file_suffix)
        print(
            f"First player won {100*np.count_nonzero(wins[wins == 1])/wins.shape[0]:.2f}% of the games!"
        )
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_value = y_train_value
        self.states = states
