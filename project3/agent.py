import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from config import Config


class Agent:

    def __init__(self, gamma, lr, input_shape) -> None:
        self.gamma = gamma

        self.train_x = [[], [], []]
        self.train_y = [[], [], []]

        input_layer = keras.Input(shape=input_shape, name="Input")
        a0 = layers.Dense(1, activation="linear")(input_layer)
        a1 = layers.Dense(1, activation="linear")(input_layer)
        a2 = layers.Dense(1, activation="linear")(input_layer)

        # Separate model for each action output. Simplifies training of Q(s,a)
        m0 = keras.Model(input_layer, a0)
        m1 = keras.Model(input_layer, a1)
        m2 = keras.Model(input_layer, a2)

        model = keras.Model(
            inputs=input_layer,
            outputs=[a0, a1, a2],
            name="Frankenstein",
        )

        m0.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )
        m1.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )
        m2.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )
        # A network with no hidden layers actually works for this problem when the input representation is designed correctly.
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.model = model

    def select_action(self, state, use_cache=False, epsilon=0.05):

        if epsilon > random.random():
            return random.randint(0, 2)

        if use_cache:  # Use already calculated value from update()
            return self.new_a
        pred = self.model(np.array([state]))
        return np.argmax(pred)

    def update(self, state=None, new_state=None, action: int = None, reward: float = 0, final=False):
        """
        s' = the state attained when doing action a in state s.
        a' = the action selected (by the current policy) to perform in state s'.
        r = the reward received in going from state s to s'
        γ = the standard RL discount factor
        """
        # x = (s,a)
        # y = r + γ * Q(s',a')
        # self.train_x.append(x)
        # self.train_y.append(y)

        x = state.copy()

        if not final:
            pred = self.model(np.array([new_state.copy()]))
            # Store value to use in next action select
            self.new_a = np.argmax(pred)
            y = reward + self.gamma * \
                np.max(pred)
        else:
            y = reward

        self.train_x[action].append(x)
        self.train_y[action].append(np.array([y]))

    def update_weights(self, epochs=1, batch_size=16):
        x0, x1, x2 = np.array(self.train_x[0]), np.array(
            self.train_x[1]), np.array(self.train_x[2])
        y0, y1, y2 = np.array(self.train_y[0]), np.array(
            self.train_y[1]), np.array(self.train_y[2])
        if x0.any():
            self.m0.fit(x0, y0, epochs=epochs, batch_size=batch_size)
        if x1.any():
            self.m1.fit(x1, y1, epochs=epochs, batch_size=batch_size)
        if x2.any():
            self.m2.fit(x2, y2, epochs=epochs, batch_size=batch_size)

    def reset_episode(self):
        # i = Config.AgentConfig.buffer_size
        # self.train_x[0] = self.train_x[0][-i:]
        # self.train_x[1] = self.train_x[1][-i:]
        # self.train_x[2] = self.train_x[2][-i:]
        # self.train_y[0] = self.train_y[0][-i:]
        # self.train_y[1] = self.train_y[1][-i:]
        # self.train_y[2] = self.train_y[2][-i:]
        self.train_x = [[], [], []]
        self.train_y = [[], [], []]
