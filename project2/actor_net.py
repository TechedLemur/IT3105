from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random


class ActorNet:
    """
    Neural network actor
    """

    def __init__(self, input_shape, output_dim) -> None:

        # TODO: maybe try convolutional later.

        input_layer = keras.Input(shape=input_shape, name="Input")

        x = layers.Dense(10, activation="relu")(input_layer)

        x = layers.Dense(12, activation="relu")(x)

        output_layer = layers.Dense(output_dim, activation=tf.nn.softmax)(x)

        self.model = keras.Model(input_layer, output_layer, name="ANET")

        self.model.compile(
            optimizer="adam",
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def select_action(self, world: GameWorld) -> Action:
        """
        Select an action based on the state.
        """

        # Softmax output
        # Rescale

        legal_actions = world.get_legal_actions()

        # While working on MCTS, just return random action
        # return random.choice(legal_actions)

        all_actions = world.get_all_actions()
        # A list with all possible actions in the game (legal and not)
        probs = self.model.predict(np.array([world.as_vector()]))

        mask = np.array([a in legal_actions for a in all_actions]).astype(np.float32)

        probs *= mask

        probs = probs / np.sum(probs)

        new_action_index = np.argmax(probs)

        # TODO: Decide if we just return the 1-hot index or the actual action
        new_action = all_actions[new_action_index]
        return new_action

    def train(self, x_train: np.array, y_train: np.array):
        self.model.fit(x=x_train, y=y_train)

    def save_params(self, i):
        self.model.save_weights(f"models/model{i}")

    def load_params(self, i):
        self.model.load_weights(f"models/model{i}")
