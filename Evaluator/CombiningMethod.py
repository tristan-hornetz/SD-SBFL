from abc import abstractmethod
from typing import Tuple, Any, Iterable, Callable, List
from .RankerEvent import EventContainer


class CombiningMethod:
    @abstractmethod
    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        pass


def avg(cs):
    return sum(cs) / len(cs)


def inv_avg(cs):
    return 1 - avg(cs)


class GenericCombiningMethod(CombiningMethod):
    def __init__(self, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        for e in events:
            coefficients.append(similarity_coefficient.compute(e))
        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),


class FilteredCombiningMethod(CombiningMethod):
    def __init__(self, event_types, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        self.event_types = event_types

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        for e in filter(lambda c: type(c) in self.event_types, events):
            coefficients.append(similarity_coefficient.compute(e))
        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),


class WeightedCombiningMethod(CombiningMethod):
    def __init__(self, weights: Iterable[Tuple[Any, float]], *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        self.weight_sum = sum(e[1] for e in weights)
        self.weights = {e[0]: e[1] / self.weight_sum for e in weights}

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        weighted_types = list(self.weights.keys())
        for e in filter(lambda c: type(c) in weighted_types, events):
            c = similarity_coefficient.compute(e)
            coefficients.append(c * self.weights[type(e)])

        if len(coefficients) == 0:
            return *([0] * len(self.methods)),
        return *(m(coefficients) for m in self.methods),


class TypeOrderCombiningMethod(GenericCombiningMethod):
    def __init__(self, types: List[type], *methods: Callable[[Iterable[float]], float]):
        super().__init__(*methods)
        self.types = types

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = {t: [] for t in self.types}
        for e in filter(lambda c: type(c) in self.types, events):
            c = similarity_coefficient.compute(e)
            coefficients[type(e)].append(c)

        return *((*(m(cs) for m in self.methods),) if len(cs) > 0 else (*([0] * len(self.methods)), ) for t, cs in coefficients.items()),


