import math
from abc import abstractmethod
from typing import Tuple, Any, Iterable, Callable, List

import numpy

from .RankerEvent import EventContainer


class CombiningMethod:
    @abstractmethod
    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        pass


def avg(cs):
    return sum(cs) / len(cs)


def inv_avg(cs):
    return 1 - avg(cs)


def median(cs):
    return sorted(cs)[len(cs)//2] if len(cs) % 2 == 1 else sum(sorted(cs)[len(cs)//2-1:len(cs)//2+1])/2


def geometric_mean(cs):
    return numpy.prod(cs) ** (1.0/len(cs))


def harmonic_mean(cs):
    return len(cs) / sum(1.0/(c if c > 0 else .01) for c in cs)


def quadratic_mean(cs):
    return math.sqrt((1/len(cs)) * sum(c**2 for c in cs))


def stddev(cs):
    if len(cs) < 1:
        return 0
    m = avg(cs)
    return math.sqrt(sum((c-m)**2 for c in cs) / len(cs))


def make_tuple(cs):
    return *sorted(cs, reverse=True),


class GenericCombiningMethod(CombiningMethod):
    def __init__(self, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        for e in events:
            coefficients.append(similarity_coefficient.compute(e))
        if len(coefficients) == 0:
            return *(m([0]) for m in self.methods),
        return *(m(coefficients) for m in self.methods),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}"
        return out


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
            return *(m([0]) for m in self.methods),
        return *(m(coefficients) for m in self.methods),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.event_types))}"
        return out


class WeightedCombiningMethod(CombiningMethod):
    def __init__(self, weights: Iterable[Tuple[Any, float]], *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        self.weight_max = max([e[1] for e in weights])
        self.weights = {e[0]: e[1] / self.weight_max for e in weights}

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        weighted_types = list(self.weights.keys())
        for e in filter(lambda c: type(c) in weighted_types, events):
            c = similarity_coefficient.compute(e)
            coefficients.append(c * self.weights[type(e)])

        if len(coefficients) == 0:
            return *(m([0]) for m in self.methods),
        return *(m(coefficients) for m in self.methods),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nWeighted event types:{str(tuple(f'{t.__name__}: {v}' for t, v in self.weights.items()))}"
        return out


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

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.types))}"
        return out

