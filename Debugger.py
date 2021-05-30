import gzip
import inspect
import math
import os
import pickle
import types
from types import FrameType, FunctionType
from typing import Any, Optional, List, Set, Tuple

from .debuggingbook.StatisticalDebugger import CoverageCollector, OchiaiDebugger, Collector


class SFL_Results:
    def __init__(self, debugger):
        self.results = debugger.rank()
        with open(inspect.getfile(self.__init__).split("/TestWrapper/")[0] + "/TestWrapper/work_dir.info", "rt") as f:
            work_dir_base = f.readline().replace("\n", "")
        work_dir = work_dir_base + "/" + os.path.abspath(os.path.curdir).replace(work_dir_base + "/", "").split("/")[0]
        with open(work_dir + "/bugsinpy_id.info", "rt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if "=" in line:
                    setattr(self, line.split("=", 1)[0].lower(), line.split("=", 1)[1].replace("\n", ""))


class ExtendedCoverageCollector(CoverageCollector):

    def __init__(self, *args, **kwargs):
        super(ExtendedCoverageCollector, self).__init__(*args, **kwargs)
        with open(inspect.getfile(self.__init__).split("/TestWrapper/")[0] + "/TestWrapper/work_dir.info", "rt") as f:
            self.work_dir_base = f.readline().replace("\n", "")

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:

        to_exclude = ["/.pyenv/", "/TestWrapper/", "/test_", "_test.py", "WrapClass.py"]
        name = frame.f_code.co_name
        function = self.search_func(name, frame)

        if function is None:
            function = self.create_function(frame)

        if not isinstance(function, FunctionType):
            return

        function_filename = inspect.getfile(function)

        if self.work_dir_base not in function_filename:
            return

        for s in to_exclude:
            if s in function_filename:
                return

        location = (function, frame.f_lineno)
        self._coverage.add(location)

    def get_filename(self, func):
        try:
            return inspect.getfile(func)
        except TypeError:
            return "<unknown>"

    def events(self) -> Set[Tuple[str, int]]:
        return {((f"{inspect.getfile(func)}[{func.__name__}]" if isinstance(func,
                                                                            types.FunctionType) else self.get_filename(
            func) + "[<unknown>]"),
                 lineno) for func, lineno in self._coverage}


class BetterOchiaiDebugger(OchiaiDebugger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collectors = {self.FAIL: list(), self.PASS: list()}
        self.collectors_with_result = {self.FAIL: dict(), self.PASS: dict()}

    def add_collector(self, outcome: str, collector: Collector) -> Collector:
        for event in collector.events():
            if event in self.collectors_with_result[outcome].keys():
                self.collectors_with_result[outcome][event].add(collector)
            else:
                self.collectors_with_result[outcome][event] = {collector}
        return super().add_collector(outcome, collector)

    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = len(self.collectors_with_result[self.FAIL][event]) if event in self.collectors_with_result[
            self.FAIL].keys() else 0
        not_in_failed = len(self.collectors[self.FAIL]) - failed
        passed = len(self.collectors_with_result[self.PASS][event]) if event in self.collectors_with_result[
            self.PASS].keys() else 0

        try:
            return failed / math.sqrt((failed + not_in_failed) * (failed + passed))
        except ZeroDivisionError:
            return None

    def rank(self) -> List[Any]:
        """Return a list of events, sorted by suspiciousness, highest first."""
        if len(self.collectors[self.FAIL]) > 0:
            return super().rank()
        return []


class ReportingDebugger(BetterOchiaiDebugger):
    def teardown(self):
        if len(self.collectors[self.FAIL]) == 0:
            return
        if not hasattr(self, "dump_file"):
            dump_file = os.path.curdir + "/TestWrapper/results.pickle.gz"
        else:
            dump_file = self.dump_file
        if os.path.isfile(dump_file):
            os.remove(dump_file)
        with gzip.open(dump_file, "xb") as f:
            pickle.dump(SFL_Results(self), f)


debugger = ReportingDebugger(collector_class=ExtendedCoverageCollector)
