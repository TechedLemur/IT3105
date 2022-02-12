from typing import Dict, Tuple
from simworlds.gambler_world import GamblerWorld, GamblerWorldAction, GamblerWorldState
from simworlds.hanoi_world import HanoiWorld, HanoiWorldAction, generate_dictionary
from simworlds.simworld import Action, State
from simworlds.pole_world import PoleWorld, PoleWorldAction, PoleWorldState


def test_hanoi_world():
    hw = HanoiWorld()
    state = hw.get_state()
    print(state.as_one_hot())
    ha = HanoiWorldAction(0, 1)
    hw.do_action(ha)
    state, _ = hw.do_action(ha)
    print(state.as_one_hot())
    HanoiWorld.visualize_solution([state.state], 3)


def test_gambler_world():
    gw = GamblerWorld()
    # print("State: ", gw.get_state())
    print("Legal Actions: ", gw.get_legal_actions())
    print(gw.do_action(GamblerWorldAction(2)))
    # print("State: ", gw.get_state())
    # gw.visualize_individual_solutions()


def test_pole_world():
    pw = PoleWorld()
    act = pw.get_legal_actions()
    print(pw.state)
    new_state, reward = pw.do_action(act[0])
    print(new_state)
    print(pw.theta_discret[new_state.theta])
    print("new internal: ", pw.state)


if __name__ == "__main__":
    test_hanoi_world()
