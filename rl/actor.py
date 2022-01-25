from simworlds.simworld import Action
import random


class Actor():

    def __init__(self, alpha: float, gamma: float, lambda_lr: float, epsilon: float) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_lr = lambda_lr  # lambda is a taken keyword in python
        self.epsilon = epsilon
        # Option 1: grow e and pi
        self.e = {}  # Eligibility traces
        self.policy = {}
        self.actions_in_episode = set()

        # Option 2: Init e an pi as tables of S X A size, where S and A is the number of possible states and actions

    # Updates the policy and eligibility
    def update(self, td_error: float) -> None:  # Step 6

        for key in self.actions_in_episode:
            self.policy[key] += self.alpha * td_error * self.e[key]
            self.e[key] *= self.gamma * self.lambda_lr  # Discount eligibility
        self.policy()

    # Do an action based on a state and current policy
    def select_action(self, state, legal_actions) -> Action:  # Step 2,3
        # Select the actions connected to the state
        filtered = dict(
            filter(lambda x: x[0][0] == state, self.policy.items()))

        # Do a random move with the probability epsilon or if Pi(s*,a) is empty for s*
        if not filtered or random.random() <= self.epsilon:
            action = legal_actions[random.randint(0, len(legal_actions))]

            # Set Pi(s,a) = 0 for the newly discovered actions and state
            if not filtered:
                for a in legal_actions:
                    self.fill_if_new_SAP(state, a)
        else:
            # Get the action from the maximum value
            action = max(filtered, key=filtered.get)[1]

        self.actions_in_episode.add((state, action))
        # A bit unsure if it is correct to set this here. Looking at step3 in the algorithm..
        self.e[(state, action)] = 1

        return action

    def reset_episode(self):
        self.e = {x: 0 for x in self.e}

    def fill_if_new_SAP(self, state, action):
        if (state, action) not in self.policy.keys():
            self.policy[(state, action)] = 0
            self.e[(state, action)] = 0
