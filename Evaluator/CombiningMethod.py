import gzip
import math
import pickle
import traceback
from abc import abstractmethod
from typing import Tuple, Any, Iterable, Callable, List, Dict
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from .RankerEvent import *
from .Ranking import RankingInfo
from correlations import EvaluationProfile

SBFL_EVENTS = [LineCoveredEvent]
SD_EVENTS = [SDScalarPairEvent, SDBranchEvent, SDReturnValueEvent]
VALUE_EVENTS = [AbsoluteReturnValueEvent, AbsoluteScalarValueEvent]


class CombiningMethod:
    @abstractmethod
    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        pass

    def update_results(self, *args, **kwargs):
        pass


def avg(cs):
    return sum(cs) / len(cs) if len(cs) > 0 else 0


def inv_avg(cs):
    return 1 - avg(cs)


def median(cs):
    return sorted(cs)[len(cs)//2] if len(cs) % 2 == 1 else sum(sorted(cs)[len(cs)//2-1:len(cs)//2+1])/2


def geometric_mean(cs):
    return np.prod(cs) ** (1.0/len(cs))


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

    @staticmethod
    def filter_single_absolute_returns(events: List[RankerEvent]):
        absolute_returns = list(filter(lambda e: isinstance(e, AbsoluteReturnValueEvent), events))
        locations = {e.location: 0 for e in absolute_returns}
        for e in absolute_returns:
            locations[e.location] += 1
        duplicate_locations = list(p for p, _ in filter(lambda e: e[1] > 1, locations.items()))
        return list(filter(lambda e: not isinstance(e, AbsoluteReturnValueEvent) or e.location in duplicate_locations, events))


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


class LinPredCombiningMethod(CombiningMethod):
    def __init__(self, *methods: Callable[[Iterable[float]], float]):
        self.methods = methods

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients_sbfl = []
        coefficients_sd = []
        for e in filter(lambda c: type(c) in SD_EVENTS, events):
            coefficients_sd.append(similarity_coefficient.compute(e))
        for e in filter(lambda c: type(c) in SBFL_EVENTS, events):
            coefficients_sbfl.append(similarity_coefficient.compute(e))
        return *((m(coefficients_sbfl if len(coefficients_sbfl) > 0 else [0]) + m(coefficients_sd if len(coefficients_sd) > 0 else [0]) / 2.0) for m in self.methods),

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


class AdjustingWeightedCombiningMethod(CombiningMethod):
    def __init__(self, start_weights: Iterable[Tuple[Any, float]], *methods: Callable[[Iterable[float]], float]):
        self.methods = methods
        weight_max = max([e[1] for e in start_weights])
        self.types = list(e[0]for e in start_weights)
        self.weights = list(e[1] / weight_max for e in start_weights)
        self.old_weights = self.weights.copy()
        self.adjust_index = 0
        self.adjust_by = -.2
        self.current_evaluation_quality = 0
        self.processed_weights = set()

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = []
        for e in filter(lambda c: type(c) in self.types, events):
            c = similarity_coefficient.compute(e)
            coefficients.append(c * self.weights[self.types.index(type(e))])
        if len(coefficients) == 0:
            return *(m([0]) for m in self.methods),
        return *(m(coefficients) for m in self.methods),

    def get_next_adj_by(self):
        self.adjust_by *= -1
        if self.adjust_by < 0:
            self.adjust_index += 1
            if self.adjust_index % len(self.weights) == 0:
                self.adjust_by /= 2.0

    def add_weights_to_processed(self):
        self.processed_weights.add(tuple((round(w, 6)) for w in self.weights))

    def next_weights_in_processed(self):
        test_weights = self.weights.copy()
        test_weights[self.adjust_index % len(self.weights)] += self.adjust_by
        return tuple((round(w, 6)) for w in test_weights) in self.processed_weights

    def update_results(self, e, *args, **kwargs):
        old_quality = self.current_evaluation_quality
        self.current_evaluation_quality = sum(e.fraction_top_k_accurate[k] + e.avg_recall_at_k[k] + e.avg_precision_at_k[k]for k in [1, 3, 5, 10])
        nw = self.weights[self.adjust_index % len(self.weights)] + self.adjust_by
        if old_quality < self.current_evaluation_quality:
            self.weights = self.old_weights.copy()
            self.get_next_adj_by()
            nw = self.weights[self.adjust_index % len(self.weights)] + self.adjust_by
        while nw < 0 or nw > 1 or self.next_weights_in_processed():
            self.get_next_adj_by()
            nw = self.weights[self.adjust_index % len(self.weights)] + self.adjust_by
        self.old_weights = self.weights.copy()
        self.weights[self.adjust_index % len(self.weights)] += self.adjust_by
        self.add_weights_to_processed()

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nWeighted event types:{str(tuple(f'{t.__name__}: {self.weights[self.types.index(t)]}' for t in self.types))}"
        return out


class TypeOrderCombiningMethod(GenericCombiningMethod):
    def __init__(self, types: List[type], *methods: Callable[[Iterable[float]], float]):
        super().__init__(*methods)
        self.types = types
        self.include_single_absolute_returns = True

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        if not self.include_single_absolute_returns:
            events = self.filter_single_absolute_returns(events)
        coefficients = {t: [] for t in self.types}
        for e in filter(lambda c: type(c) in self.types, events):
            c = similarity_coefficient.compute(e)
            coefficients[type(e)].append(c)

        return *((*(m(cs) for m in self.methods),) if len(cs) > 0 else (*([0] * len(self.methods)), ) for t, cs in coefficients.items()),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.types))}"
        return out


class GroupedTypeOrderCombiningMethod(GenericCombiningMethod):
    def __init__(self, types: List[Tuple[type]], *methods: Callable[[Iterable[float]], float]):
        super().__init__(*methods)
        self.type_groups = types

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        coefficients = {t: [] for t in self.type_groups}
        for tg in self.type_groups:
            for e in filter(lambda c: type(c) in tg, events):
                c = similarity_coefficient.compute(e)
                coefficients[tg].append(c)
        return *(tuple(m(cs) if len(cs)>0 else 0 for cs in coefficients.values()) for m in self.methods),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(self.type_groups))}"
        return out


class CompoundCombiningMethod(GenericCombiningMethod):
    def __init__(self, sub_methods: List[CombiningMethod], *methods: Callable[[Iterable[float]], float]):
        super().__init__(*methods)
        self.sub_methods = sub_methods

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        return tuple(c.combine(program_element, event_container, similarity_coefficient) for c in self.sub_methods)

    def __str__(self):
        out = super().__str__() + "\n"
        for m in self.sub_methods:
            out += "* "
            out += "\n* ".join(str(m).split("\n"))
            out += "* \n* \n"
        return out


class SystematicCombiningMethod(GenericCombiningMethod):
    def __init__(self, types: List[type], *methods: Callable[[Iterable[float]], float]):
        super().__init__(*methods)
        self.types = types
        self.current_event_container = None
        self.current_ranking = list()
        self.pre_ranking = dict()
        self.steps = [25, 15, 10]
        self.step_weights = [1, .5, .5]

    def update_event_container(self, event_container: EventContainer, similarity_coefficient):
        if event_container == self.current_event_container:
            return
        self.current_event_container = event_container
        try:
            program_elements = list(event_container.events_by_program_element.keys())
            pre_rankings = list(list(sorted(((p, self.pre_combine(p, similarity_coefficient, self.types[i])) for p in program_elements), key=lambda p: p[1], reverse=True))[:self.steps[i]] for i in range(len(self.types)))
            base_ranking = {e: tuple([0]*len(self.methods)) for e, _ in pre_rankings[0]}
            for e, _ in pre_rankings[0]:
                for i, r in enumerate(pre_rankings):
                    f = list(filter(lambda v: v[0] == e, r))
                    adjusted_coefficient = (*(x * self.step_weights[i] for x in f[0][1]),) if len(f) > 0 else tuple([0]*len(self.methods))
                    base_ranking[e] = *(adjusted_coefficient[i] + base_ranking[e][i] for i in range(len(self.methods))),
            self.current_ranking = list(e for e, _ in sorted(base_ranking.items(), key=lambda e: e[1], reverse=True))[:10]
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

    def pre_combine(self, program_element, similarity_coefficient, t):
        events = list(filter(lambda e: type(e) == t, self.current_event_container.get_from_program_element(program_element)))
        coefficients = []
        for e in events:
            coefficients.append(similarity_coefficient.compute(e))
        if len(coefficients) == 0:
            return *(m([0]) for m in self.methods),
        return *(m(coefficients) for m in self.methods),

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        self.update_event_container(event_container, similarity_coefficient)
        if program_element in self.current_ranking:
            return 1.0/(self.current_ranking.index(program_element)+1),
        return 0,

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.types))}"
        return out


class ClassifierCombiningMethod(CombiningMethod):
    def __init__(self, datasets_train, labels, combiner_lc: CombiningMethod, combiner_nlc: CombiningMethod, ranking_infos: Dict[EventContainer, RankingInfo]):
        self.classifier = RandomForestClassifier(n_estimators=25, max_depth=4, max_features=2, random_state=42)
        self.classifier.fit(datasets_train, labels)
        self.ranking_infos = ranking_infos
        self.combiner_lc = combiner_lc
        self.combiner_nlc = combiner_nlc
        self.lc_best_buffer = dict()

    class DummyEv:
        def __init__(self, ri):
            self.ranking_infos = [ri]
            self.evaluation_metrics = None

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        if event_container not in self.lc_best_buffer.keys():
            ev = self.DummyEv(self.ranking_infos[event_container])
            data = EvaluationProfile(ev).get_datasets()
            lc_best = self.classifier.predict(data)[0]
            self.lc_best_buffer[event_container] = lc_best
        else:
            lc_best = self.lc_best_buffer[event_container]
        if lc_best:
            return self.combiner_lc.combine(program_element, event_container, similarity_coefficient)
        return self.combiner_nlc.combine(program_element, event_container, similarity_coefficient)

