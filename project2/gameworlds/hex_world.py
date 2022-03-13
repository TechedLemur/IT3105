from gameworlds.gameworld import GameWorld, State, Action
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class HexState(State):

    def __hash__(self):
        return hash(repr(self))


@dataclass
class HexAction(Action):

    def __hash__(self):
        return hash(repr(self))


class HexWorld(GameWorld):

    def __init__(self):
        pass
