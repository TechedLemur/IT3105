import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Agent:

    def __init__(self, gamma, lr, input_shape) -> None:
        self.gamma = gamma

        self.train_x = []
        self.train_y = []

        model = keras.Sequential()
        model.add(keras.Input(shape=input_shape))
        model.add(layers.Dense(1, activation="linear"))
        # A network with no hidden layers actually works for this problem when the input representation is designed correctly.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.MeanSquaredError()
        )
        self.model = model

    def update(self, s_a=None, new_state=None, reward: float = 0):
        """
        s' = the state attained when doing action a in state s.
        a' = the action selected (by the current policy) to perform in state s'.
        r = the reward received in going from state s to s'
        γ = the standard RL discount factor
        """
        # TODO:
        # x = (s,a)
        # y = r + γ * Q(s',a')
        # self.train_x.append(x)
        # self.train_y.append(y)
        pass

    def update_weights(self, epochs=1, batch_size=16):

        self.model.fit(x=np.array(self.train_x), y=np.array(
            self.train_y), epochs=epochs, batch_size=batch_size)

    def reset_episode(self):
        self.train_x = []
        self.train_y = []
