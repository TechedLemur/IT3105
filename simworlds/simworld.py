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


class SimWorld(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_legal_actions(self, state: State) -> List[Action]:
        pass

    @abstractmethod
    def do_action(self, action: Action) -> Tuple[State, int]:
        pass

    @abstractmethod
    def visualize_individual_solutions(self):
        pass

    @abstractmethod
    def visualize_learning_progress(self):
        pass
