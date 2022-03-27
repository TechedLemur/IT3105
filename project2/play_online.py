import configparser
from ActorClient import ActorClient
from actor_net import ActorNet
from gameworlds.hex_world import HexState
import numpy as np
# actor = ActorNet()

config = configparser.ConfigParser()
config.read('api_token.ini')

token = config['DEFAULT']['Token']

# Import and override the `handle_get_action` hook in ActorClient

actor = ActorNet("./data/2022-03-27T19-41-30_7x7")
# TODO: Load params
actor.load_params(19, "7x7")

warmup = actor.model(np.zeros((2, 11, 11, 9)))


class MyClient(ActorClient):
    # pass
    def handle_get_action(self, state):
        state = HexState.from_array(state)
        a = actor.select_action(state, greedy=True)  # Your logic
        row = a.row
        col = a.col
        return row, col

    # Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth=token)
    client.run(mode='league')
