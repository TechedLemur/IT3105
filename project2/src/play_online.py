import configparser
from ActorClient import ActorClient
from actor_net import ActorNet
from gameworlds.hex_world import HexState
import numpy as np
class MyClient(ActorClient):

    def handle_get_action(self, state):
        state = HexState.from_array(state)

        a = actor.select_action(state, greedy=True)
        row = a.row
        col = a.col
        return row, col

    def handle_game_start(self, start_player):
        super().handle_game_start(start_player)

if __name__ == '__main__':
    actor = ActorNet("./data/middle40")
    actor.load_params(8, "7x7")
    warmup = actor.model(np.zeros((2, 11, 11, 9)))

    config = configparser.ConfigParser()
    config.read('api_token.ini')

    token = config['DEFAULT']['Token']

    client = MyClient(auth=token)
    # client.run(mode='league')
    client.run()
