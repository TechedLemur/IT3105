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
