class Config:
    """
    Configuration class
    """

    # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
    k = 3

    mini_batch_size = 500
    replay_buffer_size = 1000

    # Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
    episodes = 10
    search_games = 300
    c = 1.0
    random_rollout_move_p = 0.1

    rollout_chance = 0.6  # Chance to use rollout and not critic
    rollout_decay = 0.9

    reward_decay = 0.4  # Decay reward of early moves

    # In the ANET, the learning rate, the number of hidden layers and neurons per layer, along with any of the
    # following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.

    input_shape = (k, k, 4)
    # input_length = k*k*4
    output_length = k*k
    layers = []
    activations = []
    learning_rate = 0.001
    epsilon = 0.3
    epsilon_decay = 0.97

    # The optimizer in the ANET, with (at least) the following options all available: Adagrad, Stochastic Gradient
    # Descent (SGD), RMSProp, and Adam.
    optimizer = "adam"

    # The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an
    # untrained net prior to episode 1, at a fixed interval throughout the training episodes.
    M = 5

    # The number of games, G, to be played between any two ANET-based agents that meet during the round-robin
    # play of the TOPP.
    G = 50

    # Nim config

    K = 4
    N = 15
