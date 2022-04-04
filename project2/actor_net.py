from gameworlds.gameworld import State, Action, GameWorld
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from gameworlds.hex_world import HexState
from tensorflow.keras import layers
import numpy as np
import random
from config import cfg
import os
from keras.layers.advanced_activations import LeakyReLU

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable gpu


class ActorNet:
    """
    Two-headed neural network representing actor and critic.
    """

    def __init__(self, path: str = ".", weight_path=None) -> None:

        self.path = path

        input_layer = keras.Input(shape=cfg.input_dim, name="Input")
        x = input_layer

        for layer in cfg.pre_block_layers:
            x = self.conv_layer(
                x, layer["filters"], layer["kernel_size"], layer["activation"], "valid"
            )

        for block in cfg.residual_blocks:
            x = self.residual_block(
                x, block["filters"], block["kernel_size"], block["activation"]
            )

        y = layers.Conv2D(1, 1, strides=1, padding="same")(x)
        y = LeakyReLU()(y)
        y = layers.BatchNormalization()(y)
        y = layers.Flatten()(y)

        for d in cfg.policy_head_dense:
            y = layers.Dense(d["neurons"], activation=d["activation"])(y)

        policy_output_layer = layers.Dense(
            cfg.output_dim, activation=tf.nn.softmax, name="policy"
        )(y)

        z = layers.Conv2D(1, 1, strides=1, padding="same")(x)
        z = LeakyReLU()(z)
        z = layers.BatchNormalization()(z)
        z = layers.Flatten()(z)
        for d in cfg.value_head_dense:
            z = layers.Dense(d["neurons"], activation=d["activation"])(z)

        value_output_layer = layers.Dense(1, activation=tf.nn.tanh, name="value")(z)

        self.policy = keras.Model(input_layer, policy_output_layer, name="policy_model")
        self.value = keras.Model(input_layer, value_output_layer, name="value_model")

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
            optimizer=cfg.optimizer(learning_rate=cfg.learning_rate),
            loss=losses,
            loss_weights=lossWeights,
            metrics=["accuracy"],
        )

        if weight_path:
            self.model.load_weights(weight_path)
            print(f"Loaded weights from {weight_path}")

        self.epsilon = cfg.epsilon

    @staticmethod
    def residual_block(x, filters: List[int], kernel_size=3, activation="leaky_rely"):
        if activation == "leaky_relu":
            y = layers.Conv2D(
                kernel_size=kernel_size, filters=filters, strides=1, padding="same"
            )(x)
            y = LeakyReLU()(y)
        else:
            y = layers.Conv2D(
                kernel_size=kernel_size,
                filters=filters,
                strides=1,
                padding="same",
                activation=activation,
            )(x)
        y = layers.BatchNormalization()(y)

        y = layers.Conv2D(
            kernel_size=kernel_size, filters=filters, strides=1, padding="same"
        )(x)

        out = layers.Add()([y, x])

        out = LeakyReLU()(out)
        out = layers.BatchNormalization()(out)

        return out

    @staticmethod
    def conv_layer(x, filters: int, kernel_size: int, activation, padding):
        if activation == "leaky_relu":
            y = layers.Conv2D(
                filters=filters, kernel_size=kernel_size, strides=1, padding=padding
            )(x)
            y = LeakyReLU()(y)
        else:
            y = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                activation=activation,
            )(x)
        y = layers.BatchNormalization()(y)
        return y

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def update_epsilon(self, n=1):
        self.epsilon *= cfg.epsilon_decay ** n
        if self.epsilon < 0.0001:
            self.epsilon = 0

    def select_action(
        self, world: State, greedy=False, argmax=False, use_expert_heuristic=False
    ) -> Action:
        """
        Select an action based on the state.
        """

        legal_actions = world.get_legal_actions()

        if use_expert_heuristic:
            h = self.expert_heuristic(world, legal_actions)

            if h:
                return legal_actions[random.choice(h)]

            if random.random() < self.epsilon and not greedy:
                return random.choice(legal_actions)

        # A list with all possible actions in the game (legal and not)
        all_actions = world.get_all_actions()

        # Softmax output
        probs = self.policy(np.array([world.as_vector()]))[0]

        mask = np.array([a in legal_actions for a in all_actions]).astype(np.float32)

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
            try:
                new_action = np.random.choice(all_actions, p=probs)
            except:
                new_action = random.choice(legal_actions)

        return new_action

    def get_policy(self, state: State) -> np.array:
        """
        Args:
            state (State): State to evaluate policy for

        Returns:
            np.array: An array of probabilities corresponding to the policy for the current legal actions.
        """
        legal_actions = state.get_legal_actions()

        prediction = self.lite_model.predict(np.array([state.as_vector()]))
        probs = prediction[0].reshape(cfg.k, cfg.k)

        p = np.zeros(len(legal_actions))

        for i, a in enumerate(legal_actions):
            p[i] = probs[a.row, a.col]

        return p / np.sum(p)

    def get_policy_and_reward(
        self, state: State, use_expert_heuristic=False
    ) -> np.array:
        """
        Get policy and reward from Actor-Critic-Net.

        Args:
            state (State): State to evaluate policy and reward for.
            use_expert_heuristic (bool, optional): If expert heuristic should be used. Defaults to False.

        Returns:
            np.array: Policy (probability of each action) and the estimated reward for the state.
        """
        legal_actions = state.get_legal_actions()

        if use_expert_heuristic:
            h = self.expert_heuristic(state, legal_actions)

            if h:
                probs = np.zeros(len(legal_actions))
                probs[h] = 1
                probs = probs / np.sum(probs)
                return probs, state.player

        prediction = self.lite_model.predict(np.array([state.as_vector()]))

        probs = prediction[0][0].reshape(cfg.k, cfg.k)
        value = prediction[1][0][0]

        p = np.zeros(len(legal_actions))

        for i, a in enumerate(legal_actions):
            p[i] = probs[a.row, a.col]

        return p / np.sum(p), value

    @staticmethod
    def expert_heuristic(state, legal_actions) -> List[int]:
        """
        Use "expert" heuristic which includes:
        * Do winning moves if there are any.
        * Stop winning moves for the opponent.
        * Save bridges.

        Returns:
            List[int]: Indices for all winning moves (if any).
        """
        winning = []
        stop_opponent = []
        save_bridge = []
        save_bridge_options = state.as_vector()[:, :, 8]
        opponent_turn = HexState(
            board=state.board,
            player=-state.player,
            k=state.k,
            is_final_state=state.is_final_state,
        )
        for i, a in enumerate(legal_actions):
            if state.do_action(a).is_final_state:
                winning.append(i)
            if opponent_turn.do_action(a).is_final_state:
                stop_opponent.append(i)
            if save_bridge_options[a.row + cfg.padding, a.col + cfg.padding]:
                save_bridge.append(i)

        if winning:
            return winning
        elif stop_opponent:
            return stop_opponent
        return save_bridge

    def train(
        self,
        x_train: np.array,
        y_train: np.array,
        y_train_value: np.array,
        epochs=5,
        batch_size=32,
    ):

        self.model.fit(
            x=x_train,
            y={"policy": y_train, "value": y_train_value},
            epochs=epochs,
            batch_size=batch_size,
        )

    def save_params(self, i, suffix=""):
        self.model.save_weights(f"{self.path}/models/model{i}{suffix}")

    def load_params(self, i, suffix=""):
        self.model.load_weights(f"{self.path}/models/model{i}{suffix}")

    def update_lite_model(self):
        self.lite_model = LiteModel.from_keras_model(self.model)


class LiteModel:
    """Lite keras-model used to speed up predictions done during MC-search.
    """
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()
        self.input_index = input_det["index"]
        self.v_output_index = output_det[0]["index"]
        self.p_output_index = output_det[1]["index"]
        self.input_shape = input_det["shape"]
        self.v_output_shape = output_det[0]["shape"]
        self.p_output_shape = output_det[1]["shape"]
        self.input_dtype = input_det["dtype"]
        self.v_output_dtype = output_det[0]["dtype"]
        self.p_output_dtype = output_det[1]["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        v_out = np.zeros((count, self.v_output_shape[1]), dtype=self.v_output_dtype)
        p_out = np.zeros((count, self.p_output_shape[1]), dtype=self.p_output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i : i + 1])
            self.interpreter.invoke()
            v_out[i] = self.interpreter.get_tensor(self.v_output_index)[0]
            p_out[i] = self.interpreter.get_tensor(self.p_output_index)[0]
        return p_out, v_out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
