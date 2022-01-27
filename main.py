from typing import Dict, Tuple
from simworlds.gambler_world import GamblerWorld, GamblerWorldAction, GamblerWorldState
from simworlds.hanoi_world import HanoiWorld, HanoiWorldAction
from simworlds.simworld import Action, State
from simworlds.pole_world import PoleWorld, PoleWorldAction, PoleWorldState

if __name__ == "__main__":
    # hw = HanoiWorld()
    # print(hw.discs)
    # print(hw.pegs)
    """ ha = HanoiWorldAction(1, 1)
    hw.do_action(ha)
    print(hw.discs)
    ha = HanoiWorldAction(3, 2)
    hw.do_action(ha)
    print(hw.discs)
    ha = HanoiWorldAction(1, 2)
    hw.do_action(ha)
    print(hw.discs) """
    # ha = HanoiWorldAction(1, 0)
    # hw.do_action(ha)
    # print(hw.discs)
    # hw.visualize_individual_solutions()
    gw = GamblerWorld()
    # print("State: ", gw.get_state())
    print("Legal Actions: ", gw.get_legal_actions())
    print(gw.do_action(GamblerWorldAction(2)))
    # print("State: ", gw.get_state())
    # gw.visualize_individual_solutions()
    """ state = GamblerWorldState(1)
    action = GamblerWorldAction(2)
    k: Dict[Tuple[State, Action] : str] = {(state, action): "alright"}
    state2 = GamblerWorldState(1)
    action2 = GamblerWorldAction(2)
    print(action2.__hash__())
    print(k[(state2, action2)])
    pw = PoleWorld()
    pw.update_state(10)
    print(pw.state) """
