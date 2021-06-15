import gzip
import inspect
import math
import os
import pickle
from typing import Any, Optional, List

from TestWrapper.root.Collector import collector_type
from TestWrapper.root.debuggingbook.StatisticalDebugger import OchiaiDebugger, Collector


class BetterOchiaiDebugger(OchiaiDebugger):
    """
    OchiaiDebugger with reduced algorithmic complexity
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collectors = {self.FAIL: list(), self.PASS: list()}
        self.collectors_with_result = {self.FAIL: dict(), self.PASS: dict()}
        self.processed_collectors = set()

    def add_collector(self, outcome: str, collector: Collector) -> Collector:
        if collector in self.processed_collectors:
            return super().add_collector(outcome, collector)
        for event in collector.events():
            if event in self.collectors_with_result[outcome].keys():
                self.collectors_with_result[outcome][event] += 1
            else:
                self.collectors_with_result[outcome][event] = 1
        return super().add_collector(outcome, collector)

    def suspiciousness(self, event: Any) -> Optional[float]:
        failed = self.collectors_with_result[self.FAIL][event] if event in self.collectors_with_result[
            self.FAIL].keys() else 0
        not_in_failed = len(self.collectors[self.FAIL]) - failed
        passed = self.collectors_with_result[self.PASS][event] if event in self.collectors_with_result[
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


class SFL_Results:
    """
    A container class extracting all relevant information about a test-run from a debugger instance.
    This is required because a debugger object itself cannot be stored with pickle.
    """
    def __init__(self, debugger:BetterOchiaiDebugger, work_dir=""):
        """
        Create a SFL_Results object from a debugger instance
        :param debugger: The debugger instance
        :param work_dir: The BugsInPy working directory (only required if non-default)
        """
        if len(debugger.collectors[debugger.FAIL]) > 0 and len(debugger.collectors[debugger.PASS]) > 0:
            self.results = debugger.rank()
        else:
            self.results = []
        self.collectors = {debugger.PASS: list(list(iter(c.events())) for c in debugger.collectors[debugger.PASS]),
                           debugger.FAIL: list(list(iter(c.events())) for c in debugger.collectors[debugger.FAIL])}
        self.collectors_with_result = debugger.collectors_with_result

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


class ReportingDebugger(BetterOchiaiDebugger):
    def teardown(self):
        """
        Dump the SFL_Results of debugger to debugger.dump_file using pickle
        """
        if len(self.collectors[self.FAIL]) == 0:
            os.system(
                f"echo \"No Failures - {len(self.collectors[self.PASS])}\" > \"" + os.path.curdir + "/TestWrapper/notice.txt\"")
            return
        if not hasattr(self, "dump_file"):
            dump_file = os.path.curdir + "/TestWrapper/results.pickle.gz"
        else:
            dump_file = self.dump_file
        if os.path.isfile(dump_file):
            os.remove(dump_file)
        with gzip.open(dump_file, "xb") as f:
            pickle.dump(SFL_Results(self), f)

debugger = ReportingDebugger(collector_class=collector_type)

