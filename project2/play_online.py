import configparser
from ActorClient import ActorClient
from actor_net import ActorNet
from gameworlds.hex_world import HexState
import numpy as np
from mcts import MCTS
from player import Player
# actor = ActorNet()

config = configparser.ConfigParser()
config.read('api_token.ini')

token = config['DEFAULT']['Token']

# Import and override the `handle_get_action` hook in ActorClient

# actor = ActorNet("./data/2022-03-28T18-29-14_7x7")
actor = ActorNet(weight_path="./97score/super_model")
# TODO: Load params
# actor.load_params(199, "7x7")

warmup = actor.model(np.zeros((2, 11, 11, 9)))
# state1 = HexState.empty_board(starting_player=1)
# state2 = HexState.empty_board(starting_player=-1)

# mcts1 = MCTS(actor, state1)
# mcts2 = MCTS(actor, state2)

# TODO: See if we can use a single tree
# mcts1.run_simulations(time_out=10000)
# mcts2.run_simulations(time_out=10000)

# player1 = Player(actor, mcts1)
# player2 = Player(actor, mcts2)


class MyClient(ActorClient):

    # pass
    def handle_get_action(self, state):
        state = HexState.from_array(state)

        # self.mcts.root = self.mcts.get_node_from_state(state)

        # self.mcts.run_simulations(time_out=500)

        # D, q = self.mcts.get_visit_counts_from_root()

        a = actor.select_action(state, greedy=True)  # Your logic
        # a = self.player.get_action(state, timeout=2000)
        # all_actions = state.get_all_actions()

        # a = all_actions[np.argmax(D)]
        # self.mcts.root = self.mcts.root.children[a]
        row = a.row
        col = a.col
        return row, col

    def handle_game_start(self, start_player):
        super().handle_game_start(start_player)

        # if start_player == 1:
        # self.player = Player(actor, mcts1.copy())

        # else:
        # self.player = Player(actor, mcts2.copy())

    # Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth=token)
    client.run(mode='league')
    # client.run()
