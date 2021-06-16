import gzip
import os
import pickle
from abc import abstractmethod
from typing import List, Tuple

from TestWrapper.root.CodeInspection import extractMethodsFromCode, getBuggyMethods, BugInfo


class Ranker:
    def __init__(self, _results, info):
        self.results = _results
        self.info = info

    @abstractmethod
    def rank(self) -> List[Tuple[Tuple[str, int], int]]:
        pass

    @abstractmethod
    def __iter__(self):
        pass


class SFL_Evaluation:
    """
    Container class for all information that is relevant for a test run
    """

    def sortResultMethods(self, work_dir):
        method_dict = dict()
        ranks = dict()
        for method in self.result_methods:
            ranks[method] = len(self.result_container.results)
            if method.file in method_dict.keys():
                if method.name in method_dict[method.file].keys():
                    method_dict[method.file][method.name].append((method.linenos, method))
                else:
                    method_dict[method.file][method.name] = [(method.linenos, method)]
            else:
                method_dict[method.file] = {method.name: [(method.linenos, method)]}
        index = -1
        for event, lineno in self.result_container.results:
            index += 1
            method_str = self.getMethodStringFromEvent(event, work_dir)
            m_arr = method_str.split("[", 1)
            if m_arr[0] in method_dict.keys():
                m_name = m_arr[1].split("]")[0]
                if m_name in method_dict[m_arr[0]].keys():
                    for linenos, method in method_dict[m_arr[0]][m_name]:
                        if lineno in linenos:
                            ranks[method] = min(index, ranks[method])
        self.result_methods.sort(key=lambda m: ranks[m])

    def getMethodStringFromEvent(self, event: str, work_dir):
        return next(reversed(event.split(" @ "))).replace(work_dir + "/", "")

    def __init__(self, result_file):
        with gzip.open(result_file) as f:
            self.result_container = pickle.load(f)

        self.result_methods = list()

        self.bug_info = BugInfo(self.result_container)
        self.buggy_methods = getBuggyMethods(self.result_container, self.bug_info)
        self.result_methods = extractMethodsFromCode(self.result_container, self.bug_info)
        self.sortResultMethods(self.bug_info.work_dir)

        self.highest_rank = len(self.result_methods)
        self.best_method = None

        for b_method in self.buggy_methods:
            current_index = -1
            for e_method in self.result_methods:
                current_index += 1
                if self.highest_rank <= current_index:
                    break
                if b_method == e_method:
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
