import random
import torch
import torch.nn as nn


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
            self.model = nn.Sequential(
                nn.Linear(inputNeurons, 64),
                nn.ReLU(),
                nn.Linear(64, 50),
                nn.ReLU(),
                nn.Linear(50, 40),
                nn.Linear(40, 1),
                # nn.Flatten(0,1)
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=0.005)
            self.loss_function = torch.nn.MSELoss()
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
        td = reward + self.gamma * self.model(torch.tensor(
            new_state.as_one_hot())).item() - self.model(torch.tensor(state.as_one_hot())).item()
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
            x = state.as_one_hot()
            y = reward + self.gamma * \
                self.model(torch.tensor(new_state.as_one_hot())).item()

            self.train_x.append(x)
            self.train_y.append(y)

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

        for x, y in zip(self.train_x, self.train_y):  # TODO?: Add minibatching
            self.optimizer.zero_grad()
            y_pred = self.model(torch.tensor(x))
            loss = self.loss_function(y_pred, torch.tensor(y))
            loss.backward()
            self.optimizer.step()

    def add_if_new_state(self, state):
        if self.isTable:
            if state not in self.V.keys():
                # Initialize V(s) with small random values
                self.V[state] = random.uniform(-1., 1.)
