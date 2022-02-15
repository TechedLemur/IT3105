from telnetlib import GA
from config import Config
from rl.actor import Actor
from rl.critic import Critic
from simworlds.gambler_world import GamblerWorld
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GamblerSimulator():

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.actor = Actor(Config.GamblerActorConfig)
        self.critic = Critic(config=Config.GamblerCriticConfig,
                             inputNeurons=Config.GamblerWorldConfig.ONE_HOT_LENGTH)
        self.world = GamblerWorld()

    def run(self, episodes=Config.MainConfig.EPISODES, verbose=False):
        for episode in range(episodes):
            if verbose:
                print("Episode", episode)
            self.world.set_initial_world_state()
            state_0 = self.world.get_state()

            self.critic.reset_episode(state_0)
            self.actor.reset_episode()

            state = state_0
            legal_actions = self.world.get_legal_actions()
            action = self.actor.select_action(state, legal_actions)

            flag = False
            for _ in range(Config.GamblerWorldConfig.MAX_STEPS):

                new_state, reward = self.world.do_action(action)  # Step 1

                legal_actions = self.world.get_legal_actions()

                if new_state.is_final_state:
                    flag = True
                else:
                    new_action = self.actor.select_action(
                        new_state, legal_actions)  # Step 2

                self.actor.update_eligibility(state, action)  # Step 3

                td = self.critic.calculate_td(
                    state, new_state, reward)  # step 4,5

                # step 6
                self.critic.update(td, state=state, new_state=new_state)
                self.actor.update(td)

                # Step 7
                action = new_action
                state = new_state

                if flag:
                    break

            if not self.critic.is_table:
                self.critic.update_weights()

            if Config.MainConfig.VISUALIZE and episode % Config.MainConfig.DELAY == 0:  # Print the last solution
                print("Episode", episode)
                self.plot_greedy_strategy()

    def plot_greedy_strategy(self):
        policy = self.actor.get_greedy_policy()
        x = []
        y = []
        for state, action in policy.items():
            x.append(state.units)
            y.append(action.units)

        x = np.array(x)
        y = np.array(y)
        sns.lineplot(x=x, y=y)
        plt.show()
