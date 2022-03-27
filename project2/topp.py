from typing import List
from actor_net import ActorNet
from gameworlds.hex_world import HexState
import time
import numpy as np


class Topp:
    """
    The Tournament of Progressive Policies
    """

    def __init__(self) -> None:
        pass

    @staticmethod
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
        if plot:
            state.plot(labels=False)
        winner = -state.player
        return winner

    @staticmethod
    def play_tournament(player1: ActorNet, player2: ActorNet, no_games: int = 50, plot=False, verbose=False):

        results = np.zeros(no_games)
        starting_player = 1
        if no_games % 2 != 0:
            raise Exception("Must be even number of games")
        for i in range(no_games//2):  # |G|/2 games of a1 red

            winner = Topp.play_single_game(
                player1, player2, starting_player, plot)
            results[i] = winner

            starting_player *= -1
        for i in range(no_games//2, no_games):  # |G|/2 games of a2 red

            winner = Topp.play_single_game(
                player2, player1, starting_player, plot)
            results[i] = -winner

            starting_player *= -1

        if verbose:
            print(
                f"Player 1 won {len(results[results > 0])} / {no_games} games")

        return results

    def play_topp(agents: List[ActorNet], no_games=50):
        results = {}
        for agent in agents:
            scores = []
            for agent2 in agents:
                if agent != agent2:

                    r = Topp.play_tournament(agent, agent2, no_games)
                    scores.append(
                        f"Won {len(results[results > 0])} / {no_games} agains {agent2.name}")

            results[agent.name] = scores

        for key, value in results.items():
            print("Results for {key}: \n")
            print(
                "=============================================================================================")
            for g in value:
                print(g)
            print(
                "=============================================================================================")
