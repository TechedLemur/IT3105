from simworlds.gambler_world import GamblerWorld, GamblerWorldAction
from simworlds.hanoi_world import HanoiWorld, HanoiWorldAction

if __name__ == "__main__":
    """ hw = HanoiWorld()
    # print(hw.discs)
    # print(hw.pegs)
    ha = HanoiWorldAction(1, 1)
    hw.do_action(ha)
    print(hw.discs)
    ha = HanoiWorldAction(3, 2)
    hw.do_action(ha)
    print(hw.discs)
    ha = HanoiWorldAction(1, 2)
    hw.do_action(ha)
    print(hw.discs)
    # ha = HanoiWorldAction(1, 0)
    # hw.do_action(ha)
    # print(hw.discs)
    hw.visualize_individual_solutions() """
    gw = GamblerWorld()
    print("State: ", gw.get_state())
    print("Legal Actions: ", gw.get_legal_actions())
    gw.do_action(GamblerWorldAction(2))
    print("State: ", gw.get_state())
    # gw.visualize_individual_solutions()
