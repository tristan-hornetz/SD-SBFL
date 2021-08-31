from typing import Iterable, Dict
from .RankerEvent import EventContainer
from .CombiningMethod import CombiningMethod
from .CodeInspection.utils import BugInfo
from .CodeInspection.Methods import getBuggyMethods, DebuggerMethod


class Ranking(Iterable):
    def __init__(self, events: EventContainer, method_objects: Dict, similarity_coefficient, combining_method: CombiningMethod, info: BugInfo, buggy_methods):
        self.info = info
        self.events = events
        self.buggy_methods = buggy_methods
        self.ranking = list()
        for element in set(method_objects.values()):
            self.ranking.append((element, combining_method.combine(element, events, similarity_coefficient)))
        self.ranking.sort(key=lambda v: (v[1], element.name, element.filename, min(element.linenos)), reverse=True)
        self.buggy_in_top_k = dict()
        self.buggy_in_ranking = list()
        assert len(buggy_methods) > 0
        assert len(method_objects) > 0

        for buggy_method in self.buggy_methods:
            for program_element, sus in self.ranking:
                if self.are_methods_equal(buggy_method, program_element):
                    self.buggy_in_ranking.append((program_element, sus))
                    break

        if len(self.buggy_in_ranking) < 1:
            self.buggy_in_ranking = [(m, (0, 0)) for m in self.buggy_methods]

        assert(len(self.ranking) > 0)
        for k in [1, 3, 5, 10]:
            self.set_evaluation_metrics(k)

    def __iter__(self):
        return iter(self.ranking)

    def are_methods_equal(self, m1: DebuggerMethod, m2: DebuggerMethod):
        return m1.name == m2.name and m1.file == m2.file and len(m1.linenos.intersection(m2.linenos)) > 0

    def get_evaluation_metrics(self, k: int):
        """
        :param k: k
        :return: TopK-Accurate?, Recall@k, Precision@k
        """
        if k not in self.buggy_in_top_k.keys():
            self.set_evaluation_metrics(k)
        return self.buggy_in_top_k[k] > 0, self.buggy_in_top_k[k] / len(self.buggy_methods), self.buggy_in_top_k[k] / k

    def set_evaluation_metrics(self, k: int):
        """
        :param k: k
        """
        top_k = self.ranking[:k]
        self.buggy_in_top_k[k] = 0
        for program_element, suspiciousness in top_k:
            for m in self.buggy_methods:
                if self.are_methods_equal(program_element, m):
                    self.buggy_in_top_k[k] += 1.0

class RankingInfo:
    def __init__(self, ranking: Ranking):
        self.info = ranking.info
        self.project_name = ranking.info.project_name
        self.bug_id = ranking.info.bug_id
        self.len_events = len(ranking.events)
        self.len_methods = len(ranking.ranking)
        self.buggy_in_ranking = len(ranking.buggy_in_ranking)
        self.num_buggy_methods = len(ranking.buggy_methods)
        self.evaluation_metrics = {k: ranking.get_evaluation_metrics(k) for k in [1, 3, 5, 10]}
        self.top_10_suspiciousness_values = list(s for e, s in ranking.ranking[:10])
        self.top_10_suspiciousness_value_ties = len(self.top_10_suspiciousness_values) - len(set(self.top_10_suspiciousness_values))


class MetaRanking:
    def __init__(self, events: EventContainer, method_objects: Dict, info: BugInfo, _results):
        self._results = _results
        self.info = info
        self.events = events
        self.method_objects = method_objects
        self.buggy_methods = getBuggyMethods(_results, info)

    def rank(self, similarity_coefficient, combining_method: CombiningMethod):
        return Ranking(self.events, self.method_objects, similarity_coefficient, combining_method, self.info, self.buggy_methods)

