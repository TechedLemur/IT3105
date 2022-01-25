import random


class Critic():

    def __init__(self, isTableNotNeural: bool, alpha: float, gamma: float, lambda_lr) -> None:

        # True if table based, false if using ANN function approximation
        self.isTableNotNeural = isTableNotNeural
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_lr = lambda_lr
        if isTableNotNeural:
            # Init table based
            self.V = {}
        else:
            # Init neural network
            pass

        self.e = {}  # Eligibility dictionary

        self.current_states = set()

    def calculate_td(self, state, new_state, reward: float) -> float:  # Step 4,5
        self.add_if_new_state(new_state)
        self.e[state] = 1
        return reward + self.gamma * self.V[new_state] - self.V[state]

    # Updates V(s) and eligibility traces
    def update(self, td: float):  # Step 6
        for state in self.current_states:
            self.V[state] += self.alpha * td * self.e[state]
            self.e[state] *= self.gamma*self.lambda_lr

    # Resets the critic to be ready for new episode the state is the initial state of the system
    def reset_episode(self, state):
        self.add_if_new_state(state)
        # Set all values to 0. Maybe it also works to just re-init til dict to an empty one
        self.e = {x: 0 for x in self.e}
        self.e[state] = 1  # Set current to 1

    def add_if_new_state(self, state):
        if state not in self.V.keys():
            # Initialize V(s) with small random values
            self.V[state] = random.uniform(-1., 1.)
