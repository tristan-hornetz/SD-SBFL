from abc import abstractmethod
from typing import Collection, Tuple, Any, Iterable, Callable, List
from .RankerEvent import RankerEvent, EventContainer


class CombiningMethod:
    @abstractmethod
    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        pass


def avg(cs):
    return sum(cs) / len(cs)


def inv_arg(cs):
    return 1 - avg(cs)


class GenericCombiningMethod(CombiningMethod):
    def __init__(self, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods

    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = list(e[1] for e in filter(lambda e: e[0] in events, event_ranking))
        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),


class FilteredCombiningMethod(CombiningMethod):
    def __init__(self, event_types, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        self.event_types = event_types

    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = list(e[1] for e in filter(lambda e: e[0] in events, filter(lambda c: type(c[0]) in self.event_types, event_ranking)))
        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),


class WeightedCombiningMethod(CombiningMethod):
    def __init__(self, weights: Iterable[Tuple[Any, float]], *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        self.weight_sum = sum(e[1] for e in weights)
        self.weights = {e[0]: e[1] / self.weight_sum for e in weights}

    def combine(self, program_element, event_container: EventContainer, event_ranking: Collection[Tuple[RankerEvent, Any]]):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = list(e[1] * self.weights[type(e[0])] for e in filter(lambda e: e[0] in events, filter(lambda c: type(c[0]) in self.weights.keys(), event_ranking)))
        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),

