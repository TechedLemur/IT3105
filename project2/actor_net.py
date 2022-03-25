from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from config import Config as cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # Disable gpu


class ActorNet:
    """
    Neural network actor
    """

    def __init__(self, input_shape=cfg.input_shape, output_dim=cfg.output_length) -> None:

        input_layer = keras.Input(shape=input_shape, name="Input")

        x = layers.Conv2D(64, 3, strides=1, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)

        # TODO: Add more layers? Resnet?

        y = layers.Conv2D(1, 1, strides=1, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = keras.activations.relu(y)
        y = layers.Flatten()(y)
        policy_output_layer = layers.Dense(
            output_dim, activation=tf.nn.softmax, name="policy")(y)

        z = layers.Conv2D(1, 1, strides=1, padding='same')(x)
        z = layers.BatchNormalization()(z)
        z = keras.activations.relu(z)
        z = layers.Flatten()(z)
        z = layers.Dense(50, activation='relu')(z)

        value_output_layer = layers.Dense(
            1, activation=tf.nn.tanh, name="value")(z)

        # x = layers.Dense(100, activation="relu")(input_layer)

        # x = layers.Dense(12, activation="relu")(x)

        # policy_output_layer = layers.Dense(
        #     output_dim, activation=tf.nn.softmax, name="policy")(x)
        # value_output_layer = layers.Dense(
        #     1, activation=tf.nn.tanh, name="value")(x)

        self.policy = keras.Model(
            input_layer, policy_output_layer, name="policy_model")
        self.value = keras.Model(
            input_layer, value_output_layer, name="value_model")

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

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def update_epsilon(self):
        self.epsilon *= cfg.epsilon_decay
        if self.epsilon < 0.0001:
            self.epsilon = 0

    def select_action(self, world: State, greedy=False) -> Action:
        """
        Select an action based on the state.
        """

        # Softmax output
        # Rescale

        legal_actions = world.get_legal_actions()

        h = self.winning_heuristic(world, legal_actions)

        if h:
            return h

        if random.random() < self.epsilon and not greedy:
            return random.choice(legal_actions)

        all_actions = world.get_all_actions()

        # A list with all possible actions in the game (legal and not)
        probs = self.policy(np.array([world.as_vector()]))

        mask = np.array(
            [a in legal_actions for a in all_actions]).astype(np.float32)
        # if world.player == -1:
        # probs *= mask.reshape((world.k, world.k)).T.flatten()
        # else:
        # probs *= mask
        probs *= mask

        probs = np.around(probs, 4)

        probs = probs / np.sum(probs)

        s = np.sum(probs[0])
        # Select an action based on the probability estimate
        new_action = np.random.choice(all_actions, p=probs[0])

        # new_action = all_actions[new_action_index]

        # if world.player == -1:
        # new_action = new_action.transposed()
        return new_action

    def get_action_and_reward(self, state: State, greedy=False):

        legal_actions = state.get_legal_actions()

        all_actions = state.get_all_actions()

        # A list with all possible actions in the game (legal and not)
        prediction = self.model(np.array([state.as_vector()]))
        probs = prediction[0]
        value = prediction[1][0][0].numpy()

        h = self.winning_heuristic(state, legal_actions)

        if h:
            return h, value

        if random.random() < self.epsilon and not greedy:
            action = random.choice(legal_actions)
            return action, value

        mask = np.array(
            [a in legal_actions for a in all_actions]).astype(np.float32)
        # if world.player == -1:
        # probs *= mask.reshape((world.k, world.k)).T.flatten()
        # else:
        # probs *= mask
        probs *= mask

        probs = np.around(probs, 4)

        probs = probs / np.sum(probs)

        # Select an action based on the probability estimate
        action = np.random.choice(all_actions, p=probs[0])

        # new_action = all_actions[new_action_index]

        # if world.player == -1:
        # new_action = new_action.transposed()
        return action, value

    @staticmethod
    def winning_heuristic(state, legal_actions):
        """
        Check all child states, and if we have a winning state, choose it. 
        """
        winning = []
        for a in legal_actions:
            if state.do_action(a).is_final_state:
                winning.append(a)

        if winning:
            return random.choice(winning)
        return None

    def evaluate_state(self, state: State) -> float:
        # TODO: See if it makes sense to predict multiple states at the same time

        v = self.value(np.array([state.as_vector()]))  # v has shape 1,1

        return v[0][0].numpy()

    def train(self, x_train: np.array, y_train: np.array, y_train_value: np.array):
        self.model.fit(
            x=x_train,
            y={"policy": y_train, "value": y_train_value},
            epochs=10
            # batch_size=cfg.mini_batch_size
        )
        self.update_epsilon()

    def save_params(self, i, suffix=""):
        self.model.save_weights(f"models/model{i}{suffix}")

    def load_params(self, i, suffix=""):
        self.model.load_weights(f"models/model{i}{suffix}")
