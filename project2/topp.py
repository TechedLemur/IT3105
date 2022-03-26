from actor_net import ActorNet
from gameworlds.hex_world import HexState
import time


class Topp:
    """
    The Tournament of Progressive Policies
    """

    def __init__(self) -> None:
        pass

    def play_single_game(player1: ActorNet, player2: ActorNet, starting_player=1, plot=False) -> int:
        """
        Play a single game between two actors. 
        Returns: 1 if player1 is the winner, -1 if player 2 is the winner.
        """

        state = HexState.empty_board(starting_player=starting_player)

        while not state.is_final_state:
            if plot:
                state.plot(labels=False)
                time.sleep(0.2)
            if state.player == 1:
                move = player1.select_action(state, greedy=True, argmax=False)

            else:
                move = player2.select_action(state, greedy=True, argmax=False)

            state = state.do_action(move)

        state.plot(labels=False)
        winner = -state.player
        return winner

    def play_tournament(player1: ActorNet, player2: ActorNet, no_games: int = 50, plot=False):

        for i in range(no_games):
            pass
