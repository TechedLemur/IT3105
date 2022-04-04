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
    def play_single_game(
        player1: ActorNet, player2: ActorNet, starting_player=1, plot=False
    ) -> int:
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
                move = player1.select_action(
                    state, greedy=True, argmax=False, use_expert_heuristic=False
                )

            else:
                move = player2.select_action(
                    state, greedy=True, argmax=False, use_expert_heuristic=False
                )

            state = state.do_action(move)
        if plot:
            state.plot(labels=False)
        winner = -state.player
        return winner

    @staticmethod
    def play_tournament(
        player1: ActorNet,
        player2: ActorNet,
        no_games: int = 50,
        plot=False,
        verbose=False,
    ):

        results = np.zeros(no_games)
        if no_games % 2 != 0:
            raise Exception("Must be even number of games")
        for i in range(no_games // 2):  # |G|/2 games of a1 red starting

            winner = Topp.play_single_game(
                player1, player2, starting_player=1, plot=plot
            )
            results[i] = winner

        for i in range(no_games // 2, no_games):  # |G|/2 games of a2 blue starting

            winner = Topp.play_single_game(
                player1, player2, starting_player=-1, plot=plot
            )
            results[i] = winner

        if verbose:
            print(f"Player 1 won {len(results[results > 0])} / {no_games} games")

        return results

    def play_topp(agents: List[ActorNet], no_games=50):
        results = {}
        for a in agents:
            results[a.name] = []

        played = []
        for agent in agents:
            for agent2 in agents:
                if agent != agent2 and (agent.name, agent2.name) not in played:

                    r = Topp.play_tournament(agent, agent2, no_games)
                    results[agent.name].append(
                        f"Won {len(r[r > 0])} / {no_games} agains {agent2.name}. As starting {len(r[:no_games//2][r[:no_games//2] > 0])} / {no_games//2}, as second {len(r[no_games//2:][r[no_games//2:] > 0])} / {no_games//2}"
                    )
                    results[agent2.name].append(
                        f"Won {len(r[r < 0])} / {no_games} agains {agent.name}. As starting {len(r[:no_games//2][r[:no_games//2] < 0])} / {no_games//2}, as second {len(r[no_games//2:][r[no_games//2:] < 0])} / {no_games//2}"
                    )
                    played.append((agent.name, agent2.name))
                    played.append((agent2.name, agent.name))

        for key, value in results.items():
            print(f"Results for {key}:")
            print(
                "===================================================================="
            )
            for g in value:
                print(g)
            print(
                "====================================================================\n"
            )
