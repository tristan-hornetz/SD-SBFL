import gzip
import math
import os
import pickle
import sys
from abc import abstractmethod
from typing import Tuple, Any, Optional, Iterable, Dict

from TestWrapper.root.CodeInspection import extractMethodsFromCode, getBuggyMethods, BugInfo
from TestWrapper.root.Predicates import LineCoveredPredicate, Predicate, NoPredicate, RecordedScalarPairPredicate


class Ranker:
    def __init__(self, _results, info):
        self.results = _results
        self.info = info

    @abstractmethod
    def risk_function(self, passed, failed, total_passed, total_failed) -> float:
        pass

    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = len(self.results.collectors_with_result[self.results.FAIL][event]) if event in \
                                                                                       self.results.collectors_with_result[
                                                                                           self.results.FAIL].keys() else 0
        passed = len(self.results.collectors_with_result[self.results.PASS][event]) if event in \
                                                                                       self.results.collectors_with_result[
                                                                                           self.results.PASS].keys() else 0
        total_passed = len(self.results.collectors[self.results.PASS])
        total_failed = len(self.results.collectors[self.results.FAIL])
        return self.risk_function(passed, failed, total_passed, total_failed)

    def rank(self) -> Iterable[Tuple[Tuple[str, int], float]]:
        return sorted(((event, self.suspiciousness(event)) for event in self.results.results), key=lambda t: t[1],
                      reverse=True)

    def __iter__(self):
        return iter(self.rank())


class OchiaiRanker(Ranker):
    """
    Simply rank all observed events with the Ochiai Metric
    """

    def risk_function(self, passed, failed, total_passed, total_failed) -> float:
        not_in_failed = total_failed - failed

        try:
            return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
        except ZeroDivisionError:
            return -1.0


class PredicateOchiaiRanker(OchiaiRanker):
    def __init__(self, *args, predicates: Iterable[Predicate], risk_function=None, **kwargs):
        super(PredicateOchiaiRanker, self).__init__(*args, **kwargs)
        self.predicates = predicates
        self.event_by_line = dict()
        self._risk_function = risk_function
        self.predicate_instances = dict()
        for p in predicates:
            for k, v in p.predicate_instances.items():
                self.predicate_instances[k] = v
        for predicate_tuple, results in self.predicate_instances.items():
            passed, failed = results
            filename, method_name, lineno, *other = predicate_tuple
            line = (filename, method_name, lineno)
            if line in self.event_by_line.keys():
                self.event_by_line[line][predicate_tuple] = (passed, failed)
            else:
                self.event_by_line[line] = {predicate_tuple: (passed, failed)}

    def risk_function(self, passed: Tuple[Dict[Tuple, Tuple[bool, int]], Dict[Tuple, Tuple[bool, int]]],
                      failed: Tuple[Dict[Tuple, Tuple[bool, int]], Dict[Tuple, Tuple[bool, int]]],
                      total_passed: int, total_failed: int) -> float:
        if self._risk_function:
            return self._risk_function(passed, failed, total_passed, total_failed)
        return super().risk_function(len(passed), len(failed), total_passed, total_failed)

    def rank(self) -> Iterable[Tuple[Any, float]]:
        return sorted(((event, self.suspiciousness(event)) for event, results in self.predicate_instances.items()), key=lambda t: t[1],
                      reverse=True)

    def suspiciousness(self, event: Any) -> float:
        filename, method_name, lineno, *other = event
        line = (filename, method_name, lineno)
        if line not in self.event_by_line.keys():
            return -1.0
        events = self.event_by_line[line]
        if event not in events.keys():
            return -1.0
        passed, failed = self.event_by_line[line][event]
        total_passed = len(self.results.collectors[self.results.PASS])
        total_failed = len(self.results.collectors[self.results.FAIL])
        return self.risk_function(passed, failed, total_passed, total_failed)


class SFL_Evaluation:
    """
    Container class for all information that is relevant for a test run
    """

    class MaxAvgComparator(Iterable):
        def __init__(self, e: list, prefer_max=True):
            self.e = e
            if len(e) == 0:
                self.e = [-1]
            self.max = max(self.e)
            self.avg = sum(self.e) / len(self.e)
            self.prefer_max = prefer_max

        def __lt__(self, other: Iterable):
            o = list(other)
            assert (len(o) > 0)
            max_o = max(o)
            if self.prefer_max:
                if self.max == max_o:
                    return self.avg < sum(o) / len(o)
                return self.max < max_o
            if self.avg == sum(o) / len(o):
                return self.max < max_o
            return self.avg < sum(o) / len(o)

        def __iter__(self):
            return iter(self.e)

    def sortResultMethods(self, work_dir, ranker, rank_by_max=True):
        method_dict = dict()
        for method in self.result_methods:
            setattr(method, 'suspiciousness', [])
            if method.file in method_dict.keys():
                if method.name in method_dict[method.file].keys():
                    method_dict[method.file][method.name].append((method.linenos, method))
                else:
                    method_dict[method.file][method.name] = [(method.linenos, method)]
            else:
                method_dict[method.file] = {method.name: [(method.linenos, method)]}
        index = -1
        for event_tuple, suspiciousness in ranker.rank():
            _filename, method_name, lineno, event_type, *other = event_tuple
            filename = _filename.replace(work_dir + "/", "")
            index += 1
            if filename in method_dict.keys():
                if method_name in method_dict[filename].keys():
                    for linenos, method in method_dict[filename][method_name]:
                        if lineno in linenos:
                            if hasattr(method, 'suspiciousness'):
                                method.suspiciousness.append(suspiciousness)
                            else:
                                setattr(method, 'suspiciousness', [suspiciousness])
        self.result_methods.sort(
            key=lambda m: self.MaxAvgComparator(m.suspiciousness, prefer_max=rank_by_max), reverse=True)

    def __init__(self, result_file, predicates=None, ranker_type=PredicateOchiaiRanker, rank_by_max=True):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        if predicates is None:
            predicates = [LineCoveredPredicate(self.result_container)]

        self.predicates = predicates
        self.ranker_type = ranker_type
        self.result_methods = list()
        self.rank_by_max = rank_by_max
        self.bug_info = BugInfo(self.result_container)

        print("Verifying recorded methods...")

        self.buggy_methods = getBuggyMethods(self.result_container, self.bug_info)
        self.result_methods = extractMethodsFromCode(self.result_container, self.bug_info)
        self.ranker = ranker_type(self.result_container, self.bug_info, predicates=predicates)
        self.sortResultMethods(self.bug_info.work_dir, self.ranker, rank_by_max=rank_by_max)

        self.highest_rank = len(self.result_methods)
        self.best_method = None

        for b_method in self.buggy_methods:
            current_index = -1
            for e_method in self.result_methods:
                current_index += 1
                if self.highest_rank <= current_index:
                    break
                if b_method.file == e_method.file and b_method.name == e_method.name \
                        and len(b_method.linenos.intersection(e_method.linenos)) > 0:
                    self.highest_rank = current_index
                    self.best_method = e_method
                    break

    def __str__(self):
        return f"Results for {self.result_container.project_name}, Bug {self.result_container.bug_id}\n" + \
               f"{len(self.result_container.results)} Events were recorded\n" + \
               f"Predicate Types:\n\n{os.linesep.join(p.__class__.__name__ + p.description() for p in self.predicates)}\n" + \
               f"Ranked {len(self.ranker.predicate_instances)} Predicate Instances and {len(self.result_methods)} Methods by {'Maximum' if self.rank_by_max else 'Average'} Suspiciousness\n\n" + \
               f"There {'is one buggy function' if len(self.buggy_methods) == 1 else f'are {len(self.buggy_methods)} buggy functions'} in this commit: \n" + \
               f"{os.linesep.join(list(f'    {method}' for method in self.buggy_methods))}\n\n" + \
               f"Most suspicious method:\n    {str(self.result_methods[0])}\n\n" + \
               f"Most suspicious buggy method: Rank #{self.highest_rank + 1}, " + \
               f"Top {(self.highest_rank + 1) * 100.0 / len(self.result_methods)}%\n" + \
               f"    {str(self.best_method)}"
