from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from config import cfg
import os
from keras.layers.advanced_activations import LeakyReLU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # Disable gpu


class ActorNet:
    """
    Neural network actor
    """

    def __init__(
        self, path: str = ".", input_shape=cfg.input_shape, output_dim=cfg.output_length, weight_path=None
    ) -> None:

        self.path = path
        input_layer = keras.Input(shape=input_shape, name="Input")
        if cfg.padding:
            x = layers.Conv2D(64, kernel_size=(3+cfg.padding), strides=1,
                              padding='valid')(input_layer)
        else:
            x = layers.Conv2D(64, 3, strides=1, padding='same')(input_layer)
        x = LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(100)(x)
        # x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
        # x = keras.activations.relu(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Conv2D(64, 3, strides=1, padding='same')(x)
        # x = keras.activations.relu(x)
        # x = layers.BatchNormalization()(x)

        # TODO: Add more layers? Resnet?

        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=64)

        y = layers.Conv2D(1, 1, strides=1, padding='same')(x)
        y = LeakyReLU()(y)
        y = layers.BatchNormalization()(y)
        y = layers.Flatten()(y)
        policy_output_layer = layers.Dense(
            output_dim, activation=tf.nn.softmax, name="policy"
        )(y)

        z = layers.Conv2D(1, 1, strides=1, padding='same')(x)
        z = LeakyReLU()(z)
        z = layers.BatchNormalization()(z)
        z = layers.Flatten()(z)
        z = layers.Dense(50, activation="relu")(z)

        value_output_layer = layers.Dense(
            1, activation=tf.nn.tanh, name="value")(z)

        self.policy = keras.Model(
            input_layer, policy_output_layer, name="policy_model")
        self.value = keras.Model(
            input_layer, value_output_layer, name="value_model")

        self.model = keras.Model(
            inputs=input_layer,
            outputs=[policy_output_layer, value_output_layer],
            name="2HNET",
        )

        losses = {
            "policy": "categorical_crossentropy",
            "value": "mse",
        }
        lossWeights = {"policy": 1.0, "value": 1.0}
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss=losses,
            loss_weights=lossWeights,
            metrics=["accuracy"],
        )

        if weight_path:
            self.model.load_weights(weight_path)
            print(f"Loaded weights from {weight_path}")

        self.epsilon = cfg.epsilon

    @staticmethod
    def residual_block(x, filters: int, kernel_size=3):
        y = layers.Conv2D(kernel_size=kernel_size,
                          filters=filters, strides=1, padding='same')(x)
        y = LeakyReLU()(y)
        y = layers.BatchNormalization()(y)

        y = layers.Conv2D(
            kernel_size=kernel_size, filters=filters, strides=1, padding="same"
        )(x)

        out = layers.Add()([y, x])

        out = LeakyReLU()(out)
        out = layers.BatchNormalization()(out)

        return out

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def update_epsilon(self, n=1):
        self.epsilon *= cfg.epsilon_decay ** n
        if self.epsilon < 0.0001:
            self.epsilon = 0

    def select_action(self, world: State, greedy=False, argmax=False) -> Action:
        """
        Select an action based on the state.
        """

        legal_actions = world.get_legal_actions()
        h = self.winning_heuristic(world, legal_actions)

        if h:
            return legal_actions[random.choice(h)]

        if random.random() < self.epsilon and not greedy:
            return random.choice(legal_actions)

        # A list with all possible actions in the game (legal and not)
        all_actions = world.get_all_actions()

        # Softmax output
        probs = self.policy(np.array([world.as_vector()]))[0]

        mask = np.array(
            [a in legal_actions for a in all_actions]).astype(np.float32)
        # if world.player == -1:
        # probs *= mask.reshape((world.k, world.k)).T.flatten()
        # else:
        # probs *= mask
        probs *= mask

        probs = probs ** cfg.anet_temperature

        probs = probs / np.sum(probs)
        # Rescale
        probs = np.around(probs, 4)
        probs = probs / np.sum(probs)

        # Select an action based on the probability estimate
        if argmax:
            new_action_index = np.argmax(probs)
            new_action = all_actions[new_action_index]
        else:
            new_action = np.random.choice(all_actions, p=probs)

        # if world.player == -1:
        # new_action = new_action.transposed()
        return new_action

    def get_policy(self, state):
        """
        Returns a list of 
        """
        legal_actions = state.get_legal_actions()

        h = self.winning_heuristic(state, legal_actions)

        if h:
            probs = np.zeros(len(legal_actions))
            probs[h] = 1
            probs = probs / np.sum(probs)
            return probs

        prediction = self.policy(np.array([state.as_vector()]))
        probs = prediction[0].numpy().reshape(cfg.k, cfg.k)

        p = np.zeros(len(legal_actions))

        for i, a in enumerate(legal_actions):
            p[i] = probs[a.row, a.col]

        return p / np.sum(p)

    def get_policy_and_reward(self, state: State, greedy=False):

        legal_actions = state.get_legal_actions()

        h = self.winning_heuristic(state, legal_actions)

        if h:
            probs = np.zeros(len(legal_actions))
            probs[h] = 1
            probs = probs / np.sum(probs)
            return probs, state.player

        prediction = self.model(np.array([state.as_vector()]))
        probs = prediction[0][0].numpy().reshape(cfg.k, cfg.k)
        value = prediction[1][0][0].numpy()

        p = np.zeros(len(legal_actions))

        for i, a in enumerate(legal_actions):
            p[i] = probs[a.row, a.col]

        return p / np.sum(p), value

    @staticmethod
    def winning_heuristic(state, legal_actions) -> List[int]:
        """
        Check all child states, and if we have a winning state, choose it.
        Returns list of indices of winning moves
        """
        winning = []

        for i, a in enumerate(legal_actions):
            if state.do_action(a).is_final_state:
                winning.append(i)

        return winning

    def evaluate_state(self, state: State) -> float:
        # TODO: See if it makes sense to predict multiple states at the same time

        v = self.value(np.array([state.as_vector()]))  # v has shape 1,1

        return v[0][0].numpy()

    def train(
        self, x_train: np.array, y_train: np.array, y_train_value: np.array, epochs=5, batch_size=128
    ):
        self.model.fit(
            x=x_train,
            y={"policy": y_train, "value": y_train_value},
            epochs=epochs,
            batch_size=batch_size
        )

    def save_params(self, i, suffix=""):
        self.model.save_weights(f"{self.path}/models/model{i}{suffix}")

    def load_params(self, i, suffix=""):
        self.model.load_weights(f"{self.path}/models/model{i}{suffix}")
