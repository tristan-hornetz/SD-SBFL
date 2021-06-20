import gzip
import inspect
import math
import os
import pickle
from typing import Any, Optional, List

from TestWrapper.root.Collector import collector_type, SharedCoverageCollector
from TestWrapper.root.debuggingbook.StatisticalDebugger import OchiaiDebugger, Collector


class NoFailuresError(Exception):
    """
    Raised when all tests that were supposed to fail actually passed
    """
    pass


class BetterOchiaiDebugger(OchiaiDebugger):
    """
    OchiaiDebugger with reduced algorithmic complexity
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collectors = {self.FAIL: list(), self.PASS: list()}
        self.collectors_with_result = {self.FAIL: dict(), self.PASS: dict()}
        self.processed_collectors = set()
        self.shared_coverage = isinstance(collector_type, SharedCoverageCollector)

    def add_collector(self, outcome: str, collector: Collector) -> Collector:
        if not self.shared_coverage:
            if collector in self.processed_collectors:
                self.collectors[outcome].append(collector)
                return collector
            for event in collector.events():
                if event in self.collectors_with_result[outcome].keys():
                    self.collectors_with_result[outcome][event].add(int(collector))
                else:
                    self.collectors_with_result[outcome][event] = {int(collector)}
        self.collectors[outcome].append(int(collector))
        return collector

    def teardown(self):
        if self.shared_coverage:
            failed = set(self.collectors[self.FAIL])
            passed = set(self.collectors[self.PASS])
            for event in collector_type.shared_coverage.keys():
                collectors_with_event = collector_type.shared_coverage[event]
                failed_collectors = set(filter(lambda c: c[0] in failed, collectors_with_event.items()))
                passed_collectors = set(filter(lambda c: c[0] in passed, collectors_with_event.items()))
                self.collectors_with_result[self.FAIL][event] = failed_collectors
                self.collectors_with_result[self.PASS][event] = passed_collectors

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
            if self.shared_coverage:
                return list(collector_type.shared_coverage.keys())
            return super().rank()
        return []


class ReportingDebugger(BetterOchiaiDebugger):
    def teardown(self):
        """
        Dump the SFL_Results of debugger to debugger.dump_file using pickle
        """
        super().teardown()
        if len(self.collectors[self.FAIL]) == 0:
            raise NoFailuresError()
        if not hasattr(self, "dump_file"):
            dump_file = os.path.curdir + "/TestWrapper/results.pickle.gz"
        else:
            dump_file = self.dump_file
        if os.path.isfile(dump_file):
            os.remove(dump_file)
        with gzip.open(dump_file, "xb") as f:
            pickle.dump(SFL_Results(self), f)


class SFL_Results:
    """
    A container class extracting all relevant information about a test-run from a debugger instance.
    This is required because a debugger object itself cannot be stored with pickle.
    """
    def __init__(self, debugger:ReportingDebugger, work_dir=""):
        """
        Create a SFL_Results object from a debugger instance
        :param debugger: The debugger instance
        :param work_dir: The BugsInPy working directory (only required if non-default)
        """
        if len(debugger.collectors[debugger.FAIL]) > 0 and len(debugger.collectors[debugger.PASS]) > 0:
            self.results = debugger.rank()
        else:
            self.results = []
        self.collectors = {debugger.PASS: debugger.collectors[debugger.PASS],
                           debugger.FAIL: debugger.collectors[debugger.FAIL]}
        self.collectors_with_event = debugger.collectors_with_result
        self.collectors_with_result = dict()
        self.FAIL = debugger.FAIL
        self.PASS = debugger.PASS
        for o in self.collectors_with_event.keys():
            self.collectors_with_result[o] = dict()
            for k in self.collectors_with_event[o].keys():
                self.collectors_with_result[o][k] = set(e[0] for e in self.collectors_with_event[o][k])

        if work_dir == "":
            split_dir = "/TestWrapper/" if "/TestWrapper/" in inspect.getfile(self.__init__) else "/_root"
            work_dir_info_file = inspect.getfile(self.__init__).split(split_dir)[0] + "/TestWrapper/work_dir.info"
            if os.path.exists(work_dir_info_file):
                with open(work_dir_info_file, "rt") as f:
                    work_dir_base = f.readline().replace("\n", "")
            else:
                work_dir_base = os.curdir.rsplit("/", 1)[0]
            self.work_dir = work_dir_base + "/" + \
                       os.path.abspath(os.path.curdir).replace(work_dir_base + "/", "").split("/")[0]
        with open(self.work_dir + "/bugsinpy_id.info", "rt") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if "=" in line:
                    setattr(self, line.split("=", 1)[0].lower(), line.split("=", 1)[1].replace("\n", ""))


debugger = ReportingDebugger(collector_class=collector_type)

