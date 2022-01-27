from typing import Dict, Tuple
from simworlds.gambler_world import GamblerWorld, GamblerWorldAction, GamblerWorldState
from simworlds.hanoi_world import HanoiWorld, HanoiWorldAction
from simworlds.simworld import Action, State
from simworlds.pole_world import PoleWorld, PoleWorldAction, PoleWorldState


def test_hanoi_world():
    hw = HanoiWorld()
    ha = HanoiWorldAction(1, 1)
    hw.do_action(ha)
    ha = HanoiWorldAction(3, 2)
    hw.do_action(ha)
    ha = HanoiWorldAction(1, 2)
    hw.do_action(ha)
    ha = HanoiWorldAction(5, 1)
    hw.do_action(ha)
    ha = HanoiWorldAction(1, 0)
    hw.do_action(ha)
    ha = HanoiWorldAction(3, 1)
    hw.do_action(ha)
    ha = HanoiWorldAction(1, 1)
    print(hw.do_action(ha))
    hw.visualize_individual_solutions()


def test_gambler_world():
    gw = GamblerWorld()
    # print("State: ", gw.get_state())
    print("Legal Actions: ", gw.get_legal_actions())
    print(gw.do_action(GamblerWorldAction(2)))
    # print("State: ", gw.get_state())
    # gw.visualize_individual_solutions()


if __name__ == "__main__":
    test_hanoi_world()
