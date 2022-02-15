from config import Config
from rl.actor import Actor
from rl.critic import Critic
from simworlds.hanoi_world import HanoiWorld
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class HanoiSimulator():

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.scores = []
        self.solutions = []
        self.actor = Actor(Config.HanoiActorConfig)
        self.critic = Critic(config=Config.HanoiCriticConfig,
                             inputNeurons=Config.HanoiWorldConfig.ONE_HOT_LENGTH)
        self.world = HanoiWorld()

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
            for _ in range(Config.HanoiWorldConfig.MAX_STEPS):

                new_state, reward = self.world.do_action(action)  # Step 1

                legal_actions = self.world.get_legal_actions()

                if new_state.is_final_state:
                    flag = True
                    self.scores.append(self.world.moves)
                    self.solutions.append(self.world.state_history)
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

        if Config.MainConfig.VISUALIZE:  # Print the last solution
            HanoiWorld.visualize_solution(self.solutions[-1])

    def print_run(self, run_no: int):
        HanoiWorld.visualize_solution(
            self.solutions[run_no])

    def do_greedy_episode(self, print_result=True):
        self.actor.set_epsilon(0)

        self.run(episodes=1)

        if print_result:
            self.print_run(-1)

    def do_greedy_strategy_no_learning(self, print_result=True):
        self.actor.set_epsilon(0)
        self.world.set_initial_world_state()
        state_0 = self.world.get_state()

        self.critic.reset_episode(state_0)
        self.actor.reset_episode()

        state = state_0
        legal_actions = self.world.get_legal_actions()
        action = self.actor.select_action(state, legal_actions)
        flag = False
        for _ in range(Config.HanoiWorldConfig.MAX_STEPS):

            new_state, reward = self.world.do_action(action)  # Step 1

            legal_actions = self.world.get_legal_actions()

            if new_state.is_final_state:
                flag = True
                self.scores.append(self.world.moves)
                self.solutions.append(self.world.state_history)
            else:
                new_action = self.actor.select_action(
                    new_state, legal_actions)  # Step 2

            self.actor.update_eligibility(state, action)  # Step 3

            # Step 7
            action = new_action
            state = new_state

            if flag:
                break
        if print_result:
            self.print_run(-1)

    def plot_learning(self):
        y = np.array(self.scores)
        sns.lineplot(data=y)
        plt.show()
