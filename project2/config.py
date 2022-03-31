import json
import sys
from tensorflow import keras


class Config:

    optimizer_dict = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "adagrad": keras.optimizers.Adagrad,
        "rmsprop": keras.optimizers.RMSprop,
    }

    def __init__(self, config_name):
        with open(f"configs/{config_name}") as f:
            c = json.load(f)

        # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
        self.k = c["k"]

        self.mini_batch_size = c["mini_batch_size"]
        self.replay_buffer_size = c["replay_buffer_size"]

        # Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
        self.episodes = c["episodes"]
        self.search_games = c["search_games"]
        self.c = c["c"]
        self.random_rollout_move_p = c["random_rollout_move_p"]

        # Chance to use rollout and not critic
        self.rollout_chance = c["rollout_chance"]
        self.rollout_decay = c["rollout_decay"]
        self.reward_decay = c["reward_decay"]  # Decay reward of early moves

        # In the ANET, the learning rate, the number of hidden layers and neurons per layer, along with any of the
        # following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
        self.padding = c["padding"]
        self.network_type = c["network_type"]
        if self.network_type == "Dense":
            self.mode = 0
            self.input_dim = 2*(self.k ** 2)
        elif self.network_type == "CNN":
            self.mode = 2
            self.input_dim = (self.k + 2 * self.padding,
            self.k + 2 * self.padding,
            c["input_planes"],)

        self.output_dim = self.k ** 2
        self.layers = c["layers"]

        self.learning_rate = c["learning_rate"]
        self.epsilon = c["epsilon"]
        self.epsilon_decay = c["epsilon_decay"]

        self.alpha = c["alpha"]  # alpha for dirichlet distribution
        self.start_temperature = c["start_temperature"]
        # Converge towards argmax
        self.temperature_increase = c["temperature_increase"]
        self.temperature_max = c["temperature_max"]
        self.anet_temperature = c["anet_temperature"]
        # The optimizer in the ANET, with (at least) the following options all available: Adagrad, Stochastic Gradient
        # Descent (SGD), RMSProp, and Adam.
        self.optimizer = Config.optimizer_dict[c["optimizer"]]

        # The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an
        # untrained net prior to episode 1, at a fixed interval throughout the training episodes.
        self.M = c["M"]

        # The number of games, G, to be played between any two ANET-based agents that meet during the round-robin
        # play of the TOPP.
        self.G = c["G"]

        self.K = 5
        self.N = 5

        self.exploration_function = c["exploration_function"]
        self.layers = c["layers"]


if sys.argv[1][-4:] == "json":
    cfg_file = sys.argv[1]
else:
    cfg_file = "demo.json"

print("Config file: ", cfg_file)
cfg = Config(cfg_file)
