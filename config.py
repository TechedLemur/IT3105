class Config:
    class HanoiWorldConfig:
        DISCS = 5  # Range [3, 5]
        PEGS = 3  # Range [2, 6]
        ONE_HOT_LENGTH = PEGS ** DISCS

    class GamblerWorldConfig:
        WIN_PROBABILITY = 0.6  # Range [0, 1.0]
        ONE_HOT_LENGTH = 101

    class PoleWorldConfig:
        POLE_LENGTH = 0.5  # Range [0.1, 1.0]
        POLE_MASS = 0.25  # Range [0.05, 0.5]
        GRAVITY = -9.81  # Range [-5, -15]
        TIMESTEP = 0.05  # Range [0.01, 0.1]
        DISCRETIZATION = 8
        ONE_HOT_LENGTH = DISCRETIZATION ** 3

    class ActorConfig:
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        EPSILON = 0.1  # Initial epsilon fro epsilon-greedy actor

    class CriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE_NOT_NEURAL = True
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        NETWORK_DIMENSIONS = [1, 2, 3]

    class MainConfig:
        EPISODES = 1000
        VISUALIZE = False
        FRAME_DELAY = 500
