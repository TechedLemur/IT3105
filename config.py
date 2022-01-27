class Config:
    class HanoiWorldConfig:
        DISCS = 3  # Range [3, 5]
        PEGS = 3  # Range [2, 6]

    class GamblerWorldConfig:
        WIN_PROBABILITY = 0.5  # Range [0, 1.0]

    class PoleWorldConfig:
        POLE_LENGTH = 0.5  # Range [0.1, 1.0]
        POLE_MASS = 0.25  # Range [0.05, 0.5]
        GRAVITY = 9.81  # Range [5, 15]
        TIMESTEP = 0.05  # Range [0.01, 0.1]

