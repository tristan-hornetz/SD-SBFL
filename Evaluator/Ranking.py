from typing import Iterable, Dict
from .RankerEvent import EventContainer
from .CombiningMethod import CombiningMethod
from .CodeInspection.utils import BugInfo
from .CodeInspection.Methods import getBuggyMethods, DebuggerMethod


class Ranking(Iterable):
    def __init__(self, events: EventContainer, method_objects: Dict, similarity_coefficient, combining_method: CombiningMethod, info: BugInfo, buggy_methods):
        self.info = info
        self.events = list()
        self.buggy_methods = buggy_methods
        for e in events:
            self.events.append((e, similarity_coefficient.compute(e)))
        self.events.sort(key=lambda v: v[1], reverse=True)
        self.ranking = list()
        for element in set(method_objects.values()):
            self.ranking.append((element, combining_method.combine(element, events, self.events)))
        self.ranking.sort(key=lambda v: v[1], reverse=True)

    def __iter__(self):
        return iter(self.ranking)

    def are_methods_equal(self, m1: DebuggerMethod, m2: DebuggerMethod):
        return m1.name == m2.name and m1.file == m2.file and len(m1.linenos.intersection(m2.linenos)) > 0

    def get_evaluation_metrics(self, k: int):
        """
        :param k: k
        :return: TopK-Accurate?, Recall@k, Precision@k
        """
        top_k = self.ranking[:k]
        buggy_in_top_k = 0
        for program_element, suspiciousness in top_k:
            for m in self.buggy_methods:
                if self.are_methods_equal(program_element, m):
                    buggy_in_top_k += 1.0
        return buggy_in_top_k > 0, buggy_in_top_k / len(self.buggy_methods), buggy_in_top_k / len(top_k)


class MetaRanking:
    def __init__(self, events: EventContainer, method_objects: Dict, _results, info: BugInfo):
        self._results = _results
        self.info = info
        self.events = events
        self.method_objects = method_objects
        self.buggy_methods = getBuggyMethods(_results, info)

    def rank(self, similarity_coefficient, combining_method: CombiningMethod):
        return Ranking(self.events, self.method_objects, similarity_coefficient, combining_method, self.info, self.buggy_methods)

