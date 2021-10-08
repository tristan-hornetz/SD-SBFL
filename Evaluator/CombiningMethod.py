import gzip
import math
import os
import pickle
import traceback
from abc import abstractmethod
from typing import Tuple, Any, Iterable, Callable, List, Dict, SupportsFloat
from sklearn.neural_network import MLPClassifier
import numpy as np

from .RankerEvent import *
from correlations import EvaluationProfile

SBFL_EVENTS = [LineCoveredEvent]
SD_EVENTS = [SDScalarPairEvent, SDBranchEvent, SDReturnValueEvent]
VALUE_EVENTS = [AbsoluteReturnValueEvent, AbsoluteScalarValueEvent]
ALL_EVENT_TYPES = [LineCoveredEvent, SDBranchEvent, AbsoluteReturnValueEvent, AbsoluteScalarValueEvent, SDScalarPairEvent, SDReturnValueEvent]

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
    return np.median(cs)


def geometric_mean(cs):
    return np.prod(cs) ** (1.0/len(cs))


def harmonic_mean(cs):
    return len(cs) / sum(1.0/(c if c > 0 else .01) for c in cs)


def quadratic_mean(cs):
    return math.sqrt((1/len(cs)) * sum(c**2 for c in cs))


def stddev(cs):
    return np.std(cs)


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
        if len(events) == 0:
            for l in event_container.events_by_program_element.keys():
                if l.name == program_element.name:
                    if l.file == program_element.file:
                        if len(l.linenos.intersection(program_element.linenos)):
                            events = list(event_container.get_from_program_element(l))
                            break
            if len(events) == 0:
                return *(m([0]) for m in self.methods),
        for e in events:
            coefficients.append(similarity_coefficient.compute(e))
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

    def combine(self, program_element: DebuggerMethod, event_container: EventContainer, similarity_coefficient):
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


class AveragingCombiningMethod(CombiningMethod):
    def __init__(self, pre_combiner: CombiningMethod, *args):
        self.pre_combiner = pre_combiner

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        return np.average(self.pre_combiner.combine(program_element, event_container, similarity_coefficient)),

    def __str__(self):
        out = f"{type(self).__name__}\nAveraged Method:\n    {(os.linesep + '    ').join(str(self.pre_combiner).split(os.linesep))}\n"
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

    def get_coefficients(self, program_element, event_container: EventContainer, similarity_coefficient):
        events = list(event_container.get_from_program_element(program_element))
        if not self.include_single_absolute_returns:
            events = self.filter_single_absolute_returns(events)
        coefficients = {t: [] for t in self.types}
        for e in filter(lambda c: type(c) in self.types, events):
            c = similarity_coefficient.compute(e)
            coefficients[type(e)].append(c)
        return coefficients

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        coefficients = self.get_coefficients(program_element, event_container, similarity_coefficient)
        return *((*(m(cs) for m in self.methods),) if len(cs) > 0 else (*([0] * len(self.methods)), ) for t, cs in coefficients.items()),

    def __str__(self):
        out = f"{type(self).__name__}\nMethods: {str(tuple(self.methods))}\nEvent types:{str(tuple(t.__name__ for t in self.types))}"
        return out


class CompensatingTypeOrderCombiningMethod(TypeOrderCombiningMethod):
    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        coefficients = self.get_coefficients(program_element, event_container, similarity_coefficient)
        for t in self.types:
            if len(coefficients[t]) == 0:
                coefficients[t] = coefficients[self.types[0]].copy()
        return *((*(m(cs) for m in self.methods),) if len(cs) > 0 else (*([0] * len(self.methods)), ) for t, cs in coefficients.items()),



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


class TwoStageCombiningMethod(CombiningMethod):
    def __init__(self, first_stage: CombiningMethod, second_stage: CombiningMethod):
        self.current_event_container = ""
        self.current_ranking = list()
        self.first_stage = first_stage
        self.second_stage = second_stage
        self.first_stage_threshold = 10

    def update_event_container(self, event_container: EventContainer, similarity_coefficient):
        if f"{event_container.project_name}{event_container.bug_id}" == self.current_event_container:
            return
        self.current_event_container = f"{event_container.project_name}{event_container.bug_id}"
        try:
            self.current_ranking = list(p for p, _ in sorted(filter(lambda e: e[1][0] > 0 if isinstance(e[1][0], SupportsFloat) else e[1][0] > tuple([0] * len(e[1][0])), ((p, self.first_stage.combine(p, event_container, similarity_coefficient)) for p in event_container.events_by_program_element.keys())), key=lambda e: e[1], reverse=True)[:self.first_stage_threshold])
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        self.update_event_container(event_container, similarity_coefficient)
        if program_element not in self.current_ranking:
            ret = (0, )
            return ret
        return self.second_stage.combine(program_element, event_container, similarity_coefficient)


class ClassifierCombiningMethod(CombiningMethod):
    def __init__(self, datasets_train, labels, first_stage: CombiningMethod):
        self.classifier = MLPClassifier(hidden_layer_sizes=32, random_state=42, max_iter=500)
        self.classifier.fit(datasets_train, labels)
        self.first_stage = first_stage
        self.threshold = np.percentile(self.classifier.predict_proba(datasets_train).T[1], 67)

    @staticmethod
    def linearizer(method: DebuggerMethod, scores: List[Tuple[float, type]], buggy: bool):
        typed_scores = {t: [] for t in ALL_EVENT_TYPES}
        for score, t in scores:
            typed_scores[t].append(score)
        all_scores = np.array([s for s, _ in scores])
        ret = [buggy, len(method.linenos), np.max(all_scores), np.average(all_scores), np.std(all_scores)]
        for t in ALL_EVENT_TYPES:
            ret.append(max(typed_scores[t] + [0]))
            ret.append(np.average(typed_scores[t]))
        return np.nan_to_num(np.array(ret), nan=0, posinf=0)

    @staticmethod
    def extract_labels(X, label_dimension_index: int):
        labels = X[label_dimension_index]
        training_data_rows = list(range(X.shape[0]))
        training_data_rows.remove(label_dimension_index)
        training_data = X[np.array(training_data_rows)]
        return training_data, labels

    def combine(self, program_element, event_container: EventContainer, similarity_coefficient):
        scores = [(similarity_coefficient.compute(e), type(e)) for e in event_container.get_from_program_element(program_element)]
        linearized = self.linearizer(program_element, scores, False)
        X, _ = self.extract_labels(linearized, 0)
        X = X.T.reshape(1, -1)
        pred_proba = self.classifier.predict_proba(X)
        print(f"{pred_proba[0][1] > self.threshold}-{pred_proba[0][1]}")
        return int(pred_proba[0][1] > self.threshold), *self.first_stage.combine(program_element, event_container, similarity_coefficient)







