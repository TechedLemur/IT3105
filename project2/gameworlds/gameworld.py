from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


class Action(ABC):
    def __hash__(self):
        return hash(repr(self))


@dataclass
class State(ABC):
    is_final_state: bool

    def __hash__(self):
        return hash(repr(self))


class GameWorld(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Action]:
        pass

    @abstractmethod
    def do_action(self, action: Action) -> State:
        pass

    def get_state(self):
        return self.state

 