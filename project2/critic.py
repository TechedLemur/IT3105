import random
import torch
import torch.nn as nn


class Critic:
    def __init__(self, config, inputNeurons) -> None:

        # True if table based, false if using ANN function approximation
        self.is_table = config.IS_TABLE
        self.alpha = config.ALPHA
        self.nn_alpha = config.NN_ALPHA
        self.gamma = config.GAMMA
        self.lambda_lr = config.LAMBDA
        if self.is_table:
            # Init table based
            self.V = {}
        else:
            # Init neural network using PyTorch
            dim = config.NETWORK_DIMENSIONS
            layers = [nn.Linear(inputNeurons, dim[0])]
            for i in range(len(dim) - 1):
                layers.append(nn.Linear(dim[i], dim[i + 1]))
            layers.append(nn.Linear(dim[-1], 1))

            self.model = nn.Sequential(*layers)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.nn_alpha)

            self.loss_function = torch.nn.MSELoss()
            self.train_x = []
            self.train_y = []

        self.e = {}  # Eligibility dictionary

        self.current_states = set()  # States in the current episode

    def calculate_td(self, state, new_state, reward: float) -> float:  # Step 4,5
        if self.is_table:
            self.add_if_new_state(new_state)
            self.e[state] = 1
            return reward + self.gamma * self.V[new_state] - self.V[state]
        # Use ANN as function approximator
        td = (
            reward
            + self.gamma * self.model(torch.tensor(new_state.as_vector())).item()
            - self.model(torch.tensor(state.as_vector())).item()
        )
        return td

    # Updates V(s) and eligibility traces
    def update(
        self, td: float, state=None, new_state=None, reward: float = 0
    ):  # Step 6
        if self.is_table:
            for state in self.current_states:
                self.V[state] += self.alpha * td * self.e[state]
                self.e[state] *= self.gamma * self.lambda_lr
        else:
            x = state.as_vector()
            y = (
                reward
                + self.gamma * self.model(torch.tensor(new_state.as_vector())).item()
            )
            # Store training data for use after the episode
            self.train_x.append(x)
            self.train_y.append(y)

    # Resets the critic to be ready for new episode, the state is the initial state of the system

    def reset_episode(self, state):
        if self.is_table:
            self.current_states = set()
            self.add_if_new_state(state)
            # Set all values to 0
            self.e = {x: 0 for x in self.e}
            self.e[state] = 1  # Set current to 1
        else:
            self.train_x = []
            self.train_y = []

    def update_weights(self):
        # Run backwards pass and update weights
        for x, y in zip(self.train_x, self.train_y):
            self.optimizer.zero_grad()
            y_pred = self.model(torch.tensor(x))
            loss = self.loss_function(y_pred, torch.tensor([y]))
            loss.backward()
            self.optimizer.step()

    # Adds the state the V(s) dictionary if it is not present
    def add_if_new_state(self, state):
        if self.is_table:
            if state not in self.V.keys():
                # Initialize V(s) with small random values
                self.V[state] = random.uniform(-1.0, 1.0)
