import configparser
from ActorClient import ActorClient
from actor_net import ActorNet
from gameworlds.hex_world import HexState
# actor = ActorNet()

config = configparser.ConfigParser()
config.read('api_token.ini')

token = config['DEFAULT']['Token']

# Import and override the `handle_get_action` hook in ActorClient

actor = ActorNet()
# TODO: Load params


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
    client.run()
