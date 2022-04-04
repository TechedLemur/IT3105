import random
from re import I
from typing import List, Set
from actor_net import ActorNet
from mcts import MCTS
from gameworlds.hex_world import HexState
from config import cfg
import numpy as np
import timeit
from multiprocessing import Pool, cpu_count
from datetime import datetime
from gameworlds.gameworld import State
from topp import Topp


class ReinforcementLearningAgent:
    """The star of the show.
    """

    def __init__(self, path: str, starting_model_path=None):
        self.anet = ActorNet(path, weight_path=starting_model_path)
        self.path = path

    @staticmethod
    def episode(params, anet: ActorNet = None) -> Set[np.array]:
        """Runs an episode (a full game) for the RLA.

        Args:
            params (tuple): (weights, starting_player, rollout_chance, epsilon). Parameters for the trainin gepisode
            anet (ActorNet, optional): ActorNet to use as actor and critic for the MCTS. Defaults to None.

        Returns:
            Set[np.array]: Training data.
        """
        t1 = timeit.default_timer()
        anet_weigths = params[0]
        starting_player = params[1]
        rollout_chance = params[2]
        epsilon = params[3]

        world = HexState.empty_board(starting_player=starting_player)
        if not anet and (cfg.random_rollout_move_p + rollout_chance) < 2:
            anet = ActorNet()
            anet.set_weights(anet_weigths)
            anet.epsilon = epsilon
            anet.update_lite_model()

        mcts = MCTS(anet, world)
        move = 1
        x_train = []
        y_train = []
        states = []
        reward_factors = []
        Q = []
        all_actions = world.get_all_actions()
        temperature = cfg.start_temperature
        while not world.is_final_state:
            # print(f"Move {move}:")
            mcts.run_simulations(rollout_chance)
            if move == 1:
                if random.random() <= 0.02:
                    D = np.zeros((cfg.k, cfg.k))
                    D[cfg.k // 2, cfg.k // 2] = 1  # Set middle move 1
                    q = 0.1  # Assuming starting player more likely to win
                    D = D.flatten()
                # player = -player
                else:
                    D, q = mcts.get_visit_counts_from_root()
            else:
                D, q = mcts.get_visit_counts_from_root()

            # print("D: ", D)
            # Decay rewards after each move
            reward_factors = [x * cfg.reward_decay for x in reward_factors]
            state = mcts.root.state
            D_matrix = D.reshape((state.k, state.k))

            # Add symmetric training cases
            ReinforcementLearningAgent.add_symmetric(
                states, x_train, y_train, reward_factors, Q, state, D, D_matrix, q
            )

            # Choose actual move (a*) based on D

            probs = D ** temperature
            probs = probs / np.sum(probs)
            # Round to avoid floating point error
            probs = np.around(probs, 4)
            probs = probs / np.sum(probs)  # Normalize
            # Select action based on distribution D

            action = np.random.choice(all_actions, p=probs)

            world = world.do_action(action)
            mcts.root = mcts.root.children[action]
            move += 1
            temperature = min(
                cfg.temperature_max, temperature * cfg.temperature_increase
            )

        t2 = timeit.default_timer()
        winner = -world.player

        print(f"Game finished - used {t2-t1} seconds")

        y_train_value = winner * np.array(reward_factors)  # * 0.5 + 0.5*np.array(Q)
        # the critic can be trained by using the score obtained at the end of each
        # actual game (i.e. episode) as the target value for backpropagation, wherein the net receives each state of the recent
        # episode as input and tries to map that state to the target (or a discounted version of the target)
        return (
            np.array(x_train),
            np.array(y_train),
            y_train_value,
            winner,
            np.array(states),
        )

    def train(self, file_suffix="", n_parallel=1, train_net=True, train_interval=1):
        """Run X episodes from config-file.
        Training consists of a loop done by running MCTS, choosing best move based on distribution and then training the
        ActorNet.

        Args:
            file_suffix (str, optional): Suffix for model-name. Defaults to "".
            n_parallel (int, optional): Number of parallel processes. Defaults to 1.
            train_net (bool, optional): If the ActorNet should be trained. Defaults to True.
            train_interval (int, optional): After how many episodes the RLA should train. Defaults to 1.
        """
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
        self.anet.update_lite_model()
        print(
            f"Running {cfg.episodes} episodes with {cfg.search_games} search games, {file_suffix}"
        )

        for ep in range(cfg.episodes):
            print(f"Episode {ep}")
            print(save_params_interval)

            if ep % save_params_interval == 0:
                print("Saved network weights")
                self.anet.save_params(ep, suffix=file_suffix)

                timestamp = datetime.now().isoformat()[:19]
                # Save data for possible later training
                timestamp = timestamp.replace(":", "-")
                with open(f"{self.path}/dataset/{timestamp}_ep_{ep}.npy", "wb") as f:
                    np.save(f, states)
                    np.save(f, y_train)
                    np.save(f, y_train_value)
                print("Saved data")

            if n > 1:  # Parallel simulations
                print(f"Running {n} games in parallel")

                weights = self.anet.get_weights()
                epsilon = self.anet.epsilon

                params = [
                    (weights, starting_player, rollout_chance, epsilon)
                    for _ in range(n)
                ]

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
            if train_net and (
                (ep % train_interval == 0 or ep == cfg.episodes) and ep > 0
            ):
                contender = ActorNet(self.anet.path)

                contender.set_weights(self.anet.get_weights())
                # Select batch from newes cases
                newest = np.arange(len(x_train))[-cfg.replay_buffer_size :]
                batch_size = min(cfg.mini_batch_size, len(x_train))
                for _ in range(cfg.train_epochs):
                    # Pick random indices
                    ind = np.random.choice(newest, batch_size, replace=False)

                    contender.train(
                        x_train[ind],
                        y_train[ind],
                        y_train_value=y_train_value[ind],
                        epochs=1,
                        batch_size=24,
                    )

                results = Topp.play_tournament(contender, self.anet, no_games=100)
                won = len(results[results > 0])
                print(f"New model won {won} of 100 games.")

                if won > 53:
                    self.anet = contender
                    self.anet.update_lite_model()
                    print("Changing model")
                else:
                    print("Using old model")

                self.anet.update_epsilon(n=n)

            rollout_chance *= cfg.rollout_decay ** n
            starting_player *= -1
        wins = np.array(wins).astype(np.float32)
        self.anet.save_params(ep, suffix=file_suffix)
        print(
            f"First player won {100*np.count_nonzero(wins[wins == 1])/wins.shape[0]:.2f}% of the games!"
        )
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_value = y_train_value
        self.states = states

    @staticmethod
    def add_symmetric(
        states: List[np.array],
        x_train: List[np.array],
        y_train: List[np.array],
        reward_factors: List[float],
        Q: List[np.float],
        state: State,
        D: np.array,
        D_matrix: np.ndarray,
        q: float,
    ):
        """Adds more training cases by generating symmetric states:
        1. State as it is
        2. Transpose board and Swap players
        3. Rotate 180 degrees
        4. Rotate 180 degrees, Transpose board and  swap players

        Args:
            states (List[np.array]): List of states.
            x_train (List[np.array]): List of policies.
            y_train (List[np.array]): List of rewards.
            reward_factors (List[float]): List of reward factors.
            Q (List[np.float]): List of Q-values.
            state (State): State to generate symmetric examples for.
            D (np.array): Policy distribution from MCTS.
            D_matrix (np.ndarray): Policy distribution reshaped.
            q (float): Current reward.
        """

        states.append(state.to_array())
        # Add training cases, and their symmetric versions
        x_train.append(state.as_vector())
        y_train.append(D)
        reward_factors.append(1)
        Q.append(q)

        # Same state, seen from opposite player
        inverted = state.inverted()
        x_train.append(inverted.as_vector())
        y_train.append(D_matrix.T.flatten())
        reward_factors.append(-1)
        Q.append(-q)
        states.append(inverted.to_array())

        rot, invRot = state.rotate180()
        # Rotate 180 degrees
        x_train.append(rot.as_vector())
        y_train.append(np.rot90(D_matrix, 2).flatten())
        reward_factors.append(1)
        Q.append(q)
        states.append(rot.to_array())

        # Rotate 180 degrees, swap players
        x_train.append(invRot.as_vector())
        y_train.append(np.rot90(D_matrix.T, 2).flatten())
        reward_factors.append(-1)
        Q.append(-q)
        states.append(invRot.to_array())
