import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import random
import timeit


class Critic():

    def __init__(self, isTable: bool, alpha: float, gamma: float, lambda_lr: float, inputNeurons: int = 10) -> None:

        # True if table based, false if using ANN function approximation
        self.isTable = isTable
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_lr = lambda_lr
        self.input_neurons = inputNeurons
        if isTable:
            # Init table based
            self.V = {}
        else:
            # Init neural network
            self.model = keras.Sequential([
                keras.Input(shape=inputNeurons),
                # layers.Dense(50, activation="relu", name="layer2"),
                layers.Dense(1, activation="relu", name="output"),
            ])
            opt = keras.optimizers.Adam(
                learning_rate=0.01
            )
            self.model.compile(loss='mse', optimizer=opt)

            self.train_x = []
            self.train_y = []

        self.e = {}  # Eligibility dictionary

        self.current_states = set()

    def calculate_td(self, state, new_state, reward: float) -> float:  # Step 4,5
        if self.isTable:
            self.add_if_new_state(new_state)
            self.e[state] = 1
            return reward + self.gamma * self.V[new_state] - self.V[state]
        # start = timeit.default_timer()
        # td = reward + self.gamma * self.model.predict(np.array([new_state.as_one_hot()]))[
        #     0][0] - self.model.predict(np.array([state.as_one_hot()]))[0][0]
        td = reward + self.gamma * self.model.predict(np.array([new_state.as_one_hot()]))[
            0][0] - self.model.predict(np.array([state.as_one_hot()]))[0][0]
        # stop = timeit.default_timer()
        # print('Time td: ', stop - start)
        return td

    # Updates V(s) and eligibility traces
    def update(self, td: float, state=None, new_state=None, reward: float = 0):  # Step 6
        if self.isTable:
            for state in self.current_states:
                self.V[state] += self.alpha * td * self.e[state]
                self.e[state] *= self.gamma*self.lambda_lr
        else:
            # start = timeit.default_timer()
            x = state.as_one_hot()
            y = reward + self.gamma * \
                self.model.predict(np.array([new_state.as_one_hot()]))[0][0]

            self.train_x.append(x)
            self.train_y.append(y)
            # stop = timeit.default_timer()
            # print('Time update: ', stop - start)
            # self.model.fit(x, np.array([y]), verbose=0)

    # Resets the critic to be ready for new episode the state is the initial state of the system

    def reset_episode(self, state):
        if self.isTable:
            self.current_states = set()
            self.add_if_new_state(state)
            # Set all values to 0. Maybe it also works to just re-init til dict to an empty one
            self.e = {x: 0 for x in self.e}
            self.e[state] = 1  # Set current to 1
        else:
            self.train_x = []
            self.train_y = []

    def update_weights(self):
        # start = timeit.default_timer()
        self.model.fit(np.array(self.train_x),
                       np.array(self.train_y), batch_size=1)
        # stop = timeit.default_timer()
        # print('Time update weights: ', stop - start)

    def add_if_new_state(self, state):
        if self.isTable:
            if state not in self.V.keys():
                # Initialize V(s) with small random values
                self.V[state] = random.uniform(-1., 1.)
