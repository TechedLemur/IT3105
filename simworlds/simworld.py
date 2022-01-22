from abc import ABC, abstractmethod
from typing import List


class Action:
    pass


class SimWorld(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Action]:
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def do_action(self, action: Action):
        pass

    @abstractmethod
    def visualize_individual_solutions(self):
        pass

    @abstractmethod
    def visualize_learning_progress(self):
        pass
