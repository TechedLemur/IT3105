from gameworlds.gameworld import State, Action
from typing import List, Tuple


class ActorNet:
    """
    Neural network actor
    """

    def __init__(self, board_size: int) -> None:

        self.board_size = board_size

        # TODO: Init neural network. Start simple, maybe try convolutional later.
        pass

    def select_action(self, state: State, legalActions: List(Action)) -> Action:
        """
        Select an action based on the state.
        """

        pass
