from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


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
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def select_action(self, world: GameWorld) -> Action:
        """
        Select an action based on the state.
        """

        # Softmax output
        # Rescale

        legal_actions = world.get_legal_actions()
        all_actions = (
            world.get_all_actions()
        )  # An array with all possible actions in the game (legal and not)

        probs = self.model.predict(world.get_state().NN_representation)

        mask = np.isin(all_actions, legal_actions)

    def train(self, x_train, y_train):

        self.model.fit(x=x_train, y=y_train)
