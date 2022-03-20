import configparser
from lib2to3.pgen2.token import tok_name
from ActorClient import ActorClient
from actor_net import ActorNet

# actor = ActorNet()

config = configparser.ConfigParser()
config.read('api_token.ini')

token = config['DEFAULT']['Token']

# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):

    pass
    # def handle_get_action(self, state):
    #     row, col = actor.get_action(state) # Your logic
    #     return row, col


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':

    client = MyClient(auth=token)
    client.run()
