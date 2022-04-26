from agent import Agent


if __name__ == '__main__':

    agent = Agent(gamma=0.95, lr=0.01, input_shape=(5))

    for episode in range(N):

        agent.reset_episode()

        world = World()

        state0 = world.state

        action0 = agent.select_action(state0, use_cache=False)

        for t in range(T):  # T= timeout

            state1, reward, final = world.do_action(action0)

            agent.update(state=state0, new_state=state1,
                         action=action0, reward=reward, final=final)

            if final:
                break

            action1 = agent.select_action(state1, use_cache=True)

            state0 = state1
            action0 = action1

        # Train model
        agent.update_weights()
