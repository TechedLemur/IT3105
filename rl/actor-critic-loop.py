# Example of imagined implementation

from actor import Actor
from critic import Critic
from simworlds.simworld import SimWorld

a = Actor(alpha=0.05, gamma=0.01, lambda_lr=0.1, epsilon=0.3)
c = Critic(isTableNotNeural=True, alpha=0.04, gamma=0.03, lambda_lr=0.9)

episodes = 50
steps = 100

state_0 = "State 0"
for episode in episodes:
    c.reset_episode(state_0)
    a.reset_episode()

    state = state_0
    legal_actions = SimWorld.get_legal_actions(state)
    action = a.select_action(state, legal_actions)

    for step in steps:

        new_state, reward = SimWorld.do_action(action)  # Step 1

        legal_actions = SimWorld.get_legal_actions(new_state)

        new_action = a.select_action(new_state, legal_actions)  # Step 2,3

        td = c.calculate_td(state, new_state, reward)  # step 4,5

        # step 6
        c.update(td)
        a.update(td)

        # Step 7
        action = new_action
        state = new_state
