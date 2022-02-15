class Config:
    class HanoiWorldConfig:
        DISCS = 3  # Range [3, 5]
        PEGS = 3  # Range [2, 6]
        ONE_HOT_LENGTH = PEGS ** DISCS
        MAX_STEPS = PEGS * DISCS * 20

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

    class HanoiActorConfig:
        ALPHA = 0.02  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.85  # Trace decay
        EPSILON = 0.8  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.98  # Epislon decay rate

    class GamblerActorConfig:
        ALPHA = 0.02  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.85  # Trace decay
        EPSILON = 0.8  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.98  # Epislon decay rate

    class PoleActorConfig:
        ALPHA = 0.02  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.85  # Trace decay
        EPSILON = 0.8  # Initial epsilon for epsilon-greedy actor
        EPSILON_DECAY = 0.98  # Epislon decay rate

    class HanoiCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = True
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        NETWORK_DIMENSIONS = [64, 50]

    class GamblerCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = True
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        NETWORK_DIMENSIONS = [64, 50]

    class PoleCriticConfig:
        # True if table based, false if using ANN function approximation
        IS_TABLE = True
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.9  # Discount rate
        LAMBDA = 0.9  # Trace decay
        NETWORK_DIMENSIONS = [64, 50]

    class MainConfig:
        EPISODES = 200
        VISUALIZE = False
        FRAME_DELAY = 100
