class Config:
    """
    Configuration class
    """

    # The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
    k = 3

    mini_batch_size = 100
    replay_buffer_size = 1000

    # Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
    episodes = 100
    search_games = 300
    c = 1.0
    random_rollout_move_p = 0.1

    # In the ANET, the learning rate, the number of hidden layers and neurons per layer, along with any of the
    # following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.

    layers = []
    activations = []
    learning_rate = 0.001

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
