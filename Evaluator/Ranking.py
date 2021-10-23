from typing import Dict

from .CodeInspection.Methods import getBuggyMethods, DebuggerMethod
from .CodeInspection.utils import BugInfo
from .CombiningMethod import CombiningMethod
from .RankerEvent import *

EVENT_TYPES = [
    LineCoveredEvent,
    SDBranchEvent,
    SDReturnValueEvent,
    SDScalarPairEvent,
    AbsoluteReturnValueEvent,
    AbsoluteScalarValueEvent,
]


class Ranking(Iterable):
    """
    Represents a Ranking for a single bug
    """

    def __init__(
        self,
        events: EventContainer,
        method_objects: Dict[Tuple[str, str, int], DebuggerMethod],
        similarity_coefficient,
        combining_method: CombiningMethod,
        info: BugInfo,
        buggy_methods,
    ):
        """
        Initializer
        :param events: The translated events to create the ranking from
        :param method_objects: The program's methods as extracted from code
        :param combining_method: The combining method to be used
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        :param info: The bug information provided by the recording framework
        """
        self.info = info
        self.events = events
        self.buggy_methods = buggy_methods
        self.similarity_coefficient = similarity_coefficient
        self.combining_method = combining_method
        self.code_statistics = None
        self.ranking = list()
        for element in set(method_objects.values()):
            self.ranking.append(
                (
                    element,
                    combining_method.combine(element, events, similarity_coefficient),
                )
            )
        self.ranking.sort(
            key=lambda v: (v[1], element.name, element.file, min(element.linenos)),
            reverse=True,
        )
        self.buggy_in_top_k = dict()
        self.buggy_in_ranking = list()
        self.buggy_method_index = dict()
        assert len(buggy_methods) > 0
        assert len(method_objects) > 0

        for buggy_method in self.buggy_methods:
            self.buggy_method_index[buggy_method] = len(self.ranking)
            for i, (program_element, sus) in enumerate(self.ranking):
                if self.are_methods_equal(buggy_method, program_element):
                    self.buggy_in_ranking.append((program_element, sus))
                    self.buggy_method_index[buggy_method] = i
                    break

        if len(self.buggy_in_ranking) < 1:
            self.buggy_in_ranking = [(m, (0, 0)) for m in self.buggy_methods]

        assert len(self.ranking) > 0
        for k in [1, 3, 5, 10]:
            self.set_evaluation_metrics(k)

    def __iter__(self):
        return iter(self.ranking)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def are_methods_equal(self, m1: DebuggerMethod, m2: DebuggerMethod) -> bool:
        """
        Check if two methods are equal
        :param m1: The first method
        :param m2: The second method
        :return: True if equal, False otherwise
        """
        return (
            m1.name == m2.name
            and m1.file == m2.file
            and len(m1.linenos.intersection(m2.linenos)) > 0
        )

    def get_evaluation_metrics(self, k: int):
        """
        :param k: k
        :return: TopK-Accurate?, Recall@k, Precision@k
        """
        self.set_evaluation_metrics(k)
        return (
            self.buggy_in_top_k[k] > 0,
            self.buggy_in_top_k[k] / len(self.buggy_methods),
            self.buggy_in_top_k[k] / k,
        )

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
                    break


class RankingInfo:
    """
    Stores information about rankings with considerably lower memory requirements than rankings themselves
    """

    def __init__(self, ranking: Ranking):
        """
        Initializer
        :param ranking: The ranking instance to store information about
        """
        self.info = ranking.info
        self.project_name = ranking.info.project_name
        self.bug_id = ranking.info.bug_id
        self.combining_method = type(ranking.combining_method), str(
            ranking.combining_method
        )
        self.similarity_coefficient = ranking.similarity_coefficient
        self.buggy_methods = ranking.buggy_methods.copy()
        self.evaluation_metrics = {
            k: ranking.get_evaluation_metrics(k) for k in [1, 3, 5, 10]
        }
        self.len_events = len(ranking.events)
        self.top_10_suspiciousness_values = list(s for e, s in ranking.ranking[:10])
        self.top_10_suspiciousness_value_ties = len(
            self.top_10_suspiciousness_values
        ) - len(set(self.top_10_suspiciousness_values))
        self.num_buggy_methods = len(ranking.buggy_methods)
        self.buggy_in_ranking = len(ranking.buggy_in_ranking)
        self.buggy_method_suspiciousness_values = dict()
        self.buggy_method_ranking_index = dict()
        for m in self.buggy_methods:
            self.buggy_method_suspiciousness_values[
                m
            ] = ranking.combining_method.combine(
                m, ranking.events, self.similarity_coefficient
            )
            self.buggy_method_ranking_index[m] = ranking.buggy_method_index[m]


class MetaRanking:
    """
    Stores all information necessary to create a ranking if given a configuration
    """

    def __init__(
        self,
        events: EventContainer,
        method_objects: Dict[Tuple[str, str, int], DebuggerMethod],
        info: BugInfo,
        _results,
    ):
        """
        Initializer
        :param events: The translated events
        :param method_objects: The methods as extracted from code
        :param info: The bug information provided by the recording framework
        :param _results: The untranslated results from the recording framework
        """
        self._results = _results
        self.info = info
        self.events = events
        self.method_objects = method_objects
        self.buggy_methods = getBuggyMethods(_results, info)

    def rank(
        self, similarity_coefficient, combining_method: CombiningMethod
    ) -> Ranking:
        """
        Create a ranking from the stored information and the given configuration
        :param combining_method: The combining method to be used
        :param similarity_coefficient: The similarity coefficient to be used. Can either be an instance or just the type.
        :return: The ranking created
        """
        self.events.project_name = self.info.project_name
        self.events.bug_id = self.info.bug_id
        return Ranking(
            self.events,
            self.method_objects,
            similarity_coefficient,
            combining_method,
            self.info,
            self.buggy_methods,
        )
