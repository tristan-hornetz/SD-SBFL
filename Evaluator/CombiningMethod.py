import math
import os
import traceback
from typing import Callable, List, SupportsFloat, Dict

import numpy as np

from .RankerEvent import *
from Evaluator.CodeInspection.Methods import DebuggerMethod

SBFL_EVENTS = [LineCoveredEvent]
SD_EVENTS = [SDScalarPairEvent, SDBranchEvent, SDReturnValueEvent]
VALUE_EVENTS = [AbsoluteReturnValueEvent, AbsoluteScalarValueEvent]
ALL_EVENT_TYPES = [
    LineCoveredEvent,
    SDBranchEvent,
    AbsoluteReturnValueEvent,
    AbsoluteScalarValueEvent,
    SDScalarPairEvent,
    SDReturnValueEvent,
]


class CombiningMethod:
    """
    Base class for combining methods
    """

    @abstractmethod
    def combine(
        self,
        program_element: DebuggerMethod,
        event_container: EventContainer,
        similarity_coefficient,
    ) -> Tuple:
        """
        Combine the similarity coefficients of all events recorded for the given program element

        :param program_element: The program element to combine for
        :param event_container: The EventContainer instance storing the bug's events
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        :return: The combined value as a tuple
        """
        pass

    def update_results(self, *args, **kwargs):
        pass


def avg(cs):
    return sum(cs) / len(cs) if len(cs) > 0 else 0


def inv_avg(cs):
    return 1 - avg(cs)


def median(cs):
    return np.median(cs)


def geometric_mean(cs):
    return np.prod(cs) ** (1.0 / len(cs))


def harmonic_mean(cs):
    return len(cs) / sum(1.0 / (c if c > 0 else 0.01) for c in cs)


def quadratic_mean(cs):
    return math.sqrt((1 / len(cs)) * sum(c ** 2 for c in cs))


def stddev(cs):
    return np.std(cs)


def make_tuple(cs):
    return (*sorted(cs, reverse=True),)


class GenericCombiningMethod(CombiningMethod):
    """
    Basic combining method, utilizing multiple aggregators
    """

    def __init__(self, *methods: Callable[[Iterable[float]], float]):
        """
        :param methods: An ordered sequence of aggregation functions to be used with the combining method
        """
        self.methods = methods

    @staticmethod
    def filter_single_absolute_returns(events: List[RankerEvent]) -> List[RankerEvent]:
        """
        Filter out absolute return value events that were only executed once

        :param events: The events to filter from
        :return: The filtered events
        """
        absolute_returns = list(
            filter(lambda e: isinstance(e, AbsoluteReturnValueEvent), events)
        )
        locations = {e.location: 0 for e in absolute_returns}
        for e in absolute_returns:
            locations[e.location] += 1
        duplicate_locations = list(
            p for p, _ in filter(lambda e: e[1] > 1, locations.items())
        )
        return list(
            filter(
                lambda e: not isinstance(e, AbsoluteReturnValueEvent)
                or e.location in duplicate_locations,
                events,
            )
        )

    def combine(
        self,
        program_element: DebuggerMethod,
        event_container: EventContainer,
        similarity_coefficient,
    ):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        if len(events) == 0:
            for l in event_container.events_by_program_element.keys():
                if l.name == program_element.name:
                    if l.file == program_element.file:
                        if len(l.linenos.intersection(program_element.linenos)):
                            events = list(event_container.get_from_program_element(l))
                            break
            if len(events) == 0:
                return (*(m([0]) for m in self.methods),)
        for e in events:
            coefficients.append(similarity_coefficient.compute(e))
        return (*(m(coefficients) for m in self.methods),)

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}"
        return out


class LinPredCombiningMethod(GenericCombiningMethod):
    def combine(
        self, program_element, event_container: EventContainer, similarity_coefficient
    ):
        events = list(event_container.get_from_program_element(program_element))
        coefficients_sbfl = []
        coefficients_sd = []
        for e in filter(lambda c: type(c) in SD_EVENTS, events):
            coefficients_sd.append(similarity_coefficient.compute(e))
        for e in filter(lambda c: type(c) in SBFL_EVENTS, events):
            coefficients_sbfl.append(similarity_coefficient.compute(e))
        return (
            *(
                (
                    m(coefficients_sbfl if len(coefficients_sbfl) > 0 else [0])
                    + m(coefficients_sd if len(coefficients_sd) > 0 else [0]) / 2.0
                )
                for m in self.methods
            ),
        )

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}"
        return out


class FilteredCombiningMethod(CombiningMethod):
    """
    Filters out unwanted event types
    """

    def __init__(
        self, event_types: List[type], *methods: Callable[[Iterable[float]], float]
    ):
        """
        :param event_types: The event types to consider
        :param methods: An ordered sequence of aggregation functions to be used with the combining method
        """
        self.methods = methods
        self.event_types = event_types

    def combine(
        self,
        program_element: DebuggerMethod,
        event_container: EventContainer,
        similarity_coefficient,
    ):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        for e in filter(lambda c: type(c) in self.event_types, events):
            coefficients.append(similarity_coefficient.compute(e))
        if len(coefficients) == 0:
            return (*(m([0]) for m in self.methods),)
        return (*(m(coefficients) for m in self.methods),)

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.event_types))}"
        return out


class AveragingCombiningMethod(CombiningMethod):
    """
    Averages the output from another combining method into a single value
    """

    def __init__(self, pre_combiner: CombiningMethod, *args):
        """
        :param pre_combiner: The combining method to average the output from
        """
        self.pre_combiner = pre_combiner

    def combine(
        self, program_element, event_container: EventContainer, similarity_coefficient
    ):
        return (
            np.average(
                self.pre_combiner.combine(
                    program_element, event_container, similarity_coefficient
                )
            ),
        )

    def __str__(self):
        out = f"{type(self).__name__}\nAveraged Method:\n    {(os.linesep + '    ').join(str(self.pre_combiner).split(os.linesep))}\n"
        return out


class WeightedCombiningMethod(CombiningMethod):
    """
    Produces modifies the suspiciousness scores of each event based on a weight for its event type
    """

    def __init__(
        self,
        weights: Iterable[Tuple[type, float]],
        *methods: Callable[[Iterable[float]], float],
    ):
        """
        :param weights: The weights for each event type
        :param methods: An ordered sequence of aggregation functions to be used with the combining method
        """
        self.methods = methods
        self.weight_max = max([e[1] for e in weights])
        self.weights = {e[0]: e[1] / self.weight_max for e in weights}

    def combine(
        self, program_element, event_container: EventContainer, similarity_coefficient
    ):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        weighted_types = list(self.weights.keys())
        for e in filter(lambda c: type(c) in weighted_types, events):
            c = similarity_coefficient.compute(e)
            coefficients.append(c * self.weights[type(e)])

        if len(coefficients) == 0:
            return (*(m([0]) for m in self.methods),)
        return (*(m(coefficients) for m in self.methods),)

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nWeighted event types:{str(tuple(f'{t.__name__}: {v}' for t, v in self.weights.items()))}"
        return out


class TypeOrderCombiningMethod(GenericCombiningMethod):
    """
    Implements event-type-based tie breaking
    """

    def __init__(self, types: List[type], *methods: Callable[[Iterable[float]], float]):
        """
        :param types: An ordered sequence of event types to consider
        :param methods: An ordered sequence of aggregation functions to be used with the combining method
        """
        super().__init__(*methods)
        self.types = types
        self.include_single_absolute_returns = True

    def get_coefficients(
        self,
        program_element: DebuggerMethod,
        event_container: EventContainer,
        similarity_coefficient,
    ) -> Dict[type, List[float]]:
        """
        Get a dictionary with all similarity coefficients for each event type

        :param program_element: The program element to combine for
        :param event_container: The EventContainer instance storing the bug's events
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        :return: A dictionary with all similarity coefficients for each event type
        """
        events = list(event_container.get_from_program_element(program_element))
        if not self.include_single_absolute_returns:
            events = self.filter_single_absolute_returns(events)
        coefficients = {t: [] for t in self.types}
        for e in filter(lambda c: type(c) in self.types, events):
            c = similarity_coefficient.compute(e)
            coefficients[type(e)].append(c)
        return coefficients

    def combine(
        self, program_element, event_container: EventContainer, similarity_coefficient
    ):
        coefficients = self.get_coefficients(
            program_element, event_container, similarity_coefficient
        )
        return (
            *(
                (*(m(cs) for m in self.methods),)
                if len(cs) > 0
                else (*([0] * len(self.methods)),)
                for t, cs in coefficients.items()
            ),
        )

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.types))}"
        return out


class TwoStageCombiningMethod(CombiningMethod):
    """
    Utilize a first-stage combining method to find the 10 highest ranking elements, then rank these with the
    second-stage combining method
    """

    def __init__(self, first_stage: CombiningMethod, second_stage: CombiningMethod):
        """
        :param first_stage: The first-stage combining method
        :param second_stage: The second-stage combining method
        """
        self.current_event_container = ""
        self.current_ranking = list()
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.first_stage_threshold = 10

    def update_event_container(
        self, event_container: EventContainer, similarity_coefficient
    ):
        """
        Update the currently buffered event container

        :param event_container: The EventContainer instance storing the bug's events
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        :return: The combined value as a tuple
        """
        if (
            f"{event_container.project_name}{event_container.bug_id}"
            == self.current_event_container
        ):
            return
        self.current_event_container = (
            f"{event_container.project_name}{event_container.bug_id}"
        )
        try:
            self.current_ranking = list(
                p
                for p, _ in sorted(
                    filter(
                        lambda e: e[1][0] > 0
                        if isinstance(e[1][0], SupportsFloat)
                        else e[1][0] > tuple([0] * len(e[1][0])),
                        (
                            (
                                p,
                                self.first_stage.combine(
                                    p, event_container, similarity_coefficient
                                ),
                            )
                            for p in event_container.events_by_program_element.keys()
                        ),
                    ),
                    key=lambda e: e[1],
                    reverse=True,
                )[: self.first_stage_threshold]
            )
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
        if hasattr(self.second_stage, "update_top_10"):
            self.second_stage.update_top_10(
                self.current_ranking, event_container, similarity_coefficient
            )

    def combine(
        self, program_element, event_container: EventContainer, similarity_coefficient
    ):
        self.update_event_container(event_container, similarity_coefficient)
        if program_element not in self.current_ranking:
            ret = (0,)
            return ret
        return self.second_stage.combine(
            program_element, event_container, similarity_coefficient
        )

    def __str__(self):
        out = f"{type(self).__name__}\n"
        for i, s in enumerate((self.first_stage, self.second_stage)):
            out += f"{i}:  "
            out += "\n    ".join(str(s).split("\n")) + "\n"
        return out
