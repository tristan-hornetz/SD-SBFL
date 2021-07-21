from abc import abstractmethod
from typing import Collection, Tuple, Any
from .RankerEvent import RankerEvent, EventContainer


class CombiningMethod:
    @abstractmethod
    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        pass


class CombineMaxThenAvg(CombiningMethod):
    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = list(e[1] for e in filter(lambda e: e[0] in events, event_ranking))
        if len(coefficients) == 0:
            return -1, -1
        return max(coefficients), sum(coefficients) / len(coefficients)


class CombineMaxThenAvgFilter(CombiningMethod):
    def __init__(self, allowed_event_types):
        self.allowed_event_types = allowed_event_types

    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        events = list(filter(lambda e: type(e) in self.allowed_event_types, event_container.get_from_program_element(program_element)))
        coefficients = list(e[1] for e in filter(lambda e: e[0] in events, event_ranking))
        if len(coefficients) == 0:
            return 0, 0
        return max(coefficients), sum(coefficients) / len(coefficients)


