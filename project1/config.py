class Config:
    class HanoiWorldConfig:
        DISCS = 3  # Range [2, 6]
        PEGS = 3  # Range [3, 5]
        VECTOR_LENGTH = (
            PEGS * DISCS * DISCS
        )  # Number of possible states is M^N (M number of pegs, N number of discs)
        MAX_STEPS = PEGS * DISCS * 20
        FINAL_REWARD = 18  # 100 for 4 discs, 18 for 3 discs

    class GamblerWorldConfig:
        WIN_PROBABILITY = 0.55  # Range [0, 1.0]
        VECTOR_LENGTH = 101
        MAX_STEPS = 200

    class PoleWorldConfig:
        POLE_LENGTH = 0.5  # Range [0.1, 1.0]
        POLE_MASS = 0.25  # Range [0.05, 0.5]
        GRAVITY = -9.81  # Range [-5, -15]
        TIMESTEP = 0.05  # Range [0.01, 0.1]
        DISCRETIZATION = 4
        VECTOR_LENGTH = 4

    class HanoiActorConfig:
        ALPHA = 0.015  # Learning rate
        GAMMA = 0.95  # Discount rate
        LAMBDA = 0.97  # Trace decay
        EPSILON = 0.5  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.988  # Epislon decay rate

    class GamblerActorConfig:
        ALPHA = 0.04  # Learning rate
        GAMMA = 0.93  # Discount rate
        LAMBDA = 0.94  # Trace decay
        EPSILON = 0.2  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.95  # Epislon decay rate

    class PoleActorConfig:
        ALPHA = 0.02  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.85  # Trace decay
        EPSILON = 0.5  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.98  # Epislon decay rate

    class HanoiCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = False
        ALPHA = 0.02  # Learning rate
        NN_ALPHA = 0.001  # Neural network LR
        GAMMA = 0.95  # Discount rate
        LAMBDA = 0.98  # Trace decay
        NETWORK_DIMENSIONS = [100, 64, 40]

    class GamblerCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = True
        ALPHA = 0.02  # Learning rate
        NN_ALPHA = 0.001  # Neural network LR
        GAMMA = 0.91  # Discount rate
        LAMBDA = 0.93  # Trace decay
        NETWORK_DIMENSIONS = [64, 40, 20]

    class PoleCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = True
        ALPHA = 0.02  # Learning rate
        NN_ALPHA = 0.001  # Neural network LR
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        NETWORK_DIMENSIONS = [128, 64, 32]

    class MainConfig:
        EPISODES = 300
        VISUALIZE = False
        DELAY = 20
