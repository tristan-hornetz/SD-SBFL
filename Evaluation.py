import gzip
import math
import os
import pickle
from abc import abstractmethod
from typing import List, Tuple, Any, Optional, Iterable

from TestWrapper.root.CodeInspection import extractMethodsFromCode, getBuggyMethods, BugInfo


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
        return sorted(((event, self.suspiciousness(event)) for event in self.results.results), key=lambda t: t[1], reverse=True)

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
    def __init__(self, *args, risk_function=None, predicates=None, **kwargs):
        super(PredicateOchiaiRanker, self).__init__(*args, **kwargs)
        if predicates is None:
            predicates = [lambda x: True]
        self.event_by_line = dict()
        self._risk_function = risk_function
        for event_tuple in self.results.results:
            filename, method_name, lineno, event_type, *other = event_tuple
            if len(self.results.collectors_with_result[self.results.FAIL][event_tuple]) > 0:
                include = False
                for p in predicates:
                    include = include or p(event_tuple)
                if not include:
                    continue
                line = (filename, method_name, lineno)
                passed = len(self.results.collectors_with_result[self.results.PASS][event_tuple])
                failed = len(self.results.collectors_with_result[self.results.FAIL][event_tuple])
                if line in self.event_by_line.keys():
                    self.event_by_line[line][event_tuple] = (passed, failed)
                else:
                    self.event_by_line[line] = {event_tuple: (passed, failed)}

    def risk_function(self, passed, failed, total_passed, total_failed) -> float:
        if self._risk_function:
            return self._risk_function(passed, failed, total_passed, total_failed)
        return super().risk_function(passed, failed, total_passed, total_failed)

    def suspiciousness(self, event: Any) -> Optional[float]:
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
        def __init__(self, e: list):
            assert(len(e) > 0)
            self.e = e
            self.max = max(*e)
            self.avg = sum(e) / len(e)

        def __lt__(self, other: Iterable):
            o = list(other)
            assert(len(o) > 0)
            max_o = max(*o)
            if self.max == max_o:
                return self.avg < sum(o) / len(o)
            return self.max < max_o

        def __iter__(self):
            return iter(self.e)

    def sortResultMethods(self, work_dir, ranker_type):
        method_dict = dict()
        for method in self.result_methods:
            if method.file in method_dict.keys():
                if method.name in method_dict[method.file].keys():
                    method_dict[method.file][method.name].append((method.linenos, method))
                else:
                    method_dict[method.file][method.name] = [(method.linenos, method)]
            else:
                method_dict[method.file] = {method.name: [(method.linenos, method)]}
        index = -1
        for event_tuple, suspiciousness in ranker_type(self.result_container, self.bug_info).rank():
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
        self.result_methods.sort(key=lambda m: self.MaxAvgComparator(m.suspiciousness) if hasattr(method, 'suspiciousness') else [-1], reverse=True)

    def __init__(self, result_file, ranker_type=PredicateOchiaiRanker):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        self.ranker_type = ranker_type

        self.result_methods = list()

        self.bug_info = BugInfo(self.result_container)
        self.buggy_methods = getBuggyMethods(self.result_container, self.bug_info)
        self.result_methods = extractMethodsFromCode(self.result_container, self.bug_info)
        self.sortResultMethods(self.bug_info.work_dir, ranker_type)

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
               f"Ranked {len(self.result_container.results)} Events and {len(self.result_methods)} Methods\n\n" + \
               f"There {'is one buggy function' if len(self.buggy_methods) == 1 else f'are {len(self.buggy_methods)} buggy functions'} in this commit: \n" + \
               f"{os.linesep.join(list(f'    {method}' for method in self.buggy_methods))}\n\n" + \
               f"Most suspicious method:\n    {str(self.result_methods[0])}\n\n" + \
               f"Most suspicious buggy method: Rank #{self.highest_rank + 1}, " + \
               f"Top {(self.highest_rank + 1) * 100.0 / len(self.result_methods)}%\n" + \
               f"    {str(self.best_method)}"
