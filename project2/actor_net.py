from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from config import Config as cfg


class ActorNet:
    """
    Neural network actor
    """

    def __init__(self, input_shape, output_dim) -> None:

        # TODO: add more layers

        input_layer = keras.Input(shape=input_shape, name="Input")

        x = layers.Dense(100, activation="relu")(input_layer)

        # x = layers.Dense(12, activation="relu")(x)

        policy_output_layer = layers.Dense(
            output_dim, activation=tf.nn.softmax, name="policy")(x)
        value_output_layer = layers.Dense(
            1, activation=tf.nn.tanh, name="value")(x)

        self.model = keras.Model(inputs=input_layer, outputs=[
                                 policy_output_layer, value_output_layer], name="2HNET")

        losses = {
            "policy": "categorical_crossentropy",
            "value": "mse",
        }
        lossWeights = {"policy": 1.0, "value": 1.0}
        self.model.compile(
            optimizer="adam",
            loss=losses,
            loss_weights=lossWeights,
            metrics=["accuracy"],
        )

        self.epsilon = cfg.epsilon

    def update_epsilon(self):
        self.epsilon *= cfg.epsilon_decay
        if self.epsilon < 0.0001:
            self.epsilon = 0

    def select_action(self, world: GameWorld) -> Action:
        """
        Select an action based on the state.
        """

        # Softmax output
        # Rescale

        legal_actions = world.get_legal_actions()

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        all_actions = world.get_all_actions()

        # A list with all possible actions in the game (legal and not)
        probs = self.model(np.array([world.as_vector()]))[0]

        mask = np.array(
            [a in legal_actions for a in all_actions]).astype(np.float32)
        if world.player == -1:
            probs *= mask.reshape((world.k, world.k)).T.flatten()
        else:
            probs *= mask

        probs = probs / np.sum(probs)

        new_action_index = np.argmax(probs)

        # TODO: Decide if we just return the 1-hot index or the actual action
        new_action = all_actions[new_action_index]

        if world.player == -1:
            new_action = new_action.transposed()
        return new_action

    def train(self, x_train: np.array, y_train: np.array, y_train_value: np.array):
        self.model.fit(
            x=x_train,
            y={"policy": y_train, "value": y_train_value}
            # batch_size=cfg.mini_batch_size
        )
        self.update_epsilon()

    def save_params(self, i, suffix=""):
        self.model.save_weights(f"models/model{i}{suffix}")

    def load_params(self, i, suffix=""):
        self.model.load_weights(f"models/model{i}{suffix}")
