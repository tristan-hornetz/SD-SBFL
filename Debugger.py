from .debuggingbook.StatisticalDebugger import CoverageCollector, OchiaiDebugger, Collector
from types import FrameType
from typing import Any, Optional, List
import inspect
import os
import math


class ExtendedCoverageCollector(CoverageCollector):

    def collect(self, frame: FrameType, event: str, arg: Any) -> None:
        filename = inspect.getfile(frame)
        to_exclude = ["/.pyenv/", "/TestWrapper/", "/test_", "_test.py"]
        if os.path.curdir in filename:
            for s in to_exclude:
                if s in filename:
                    return
            super().collect(frame, event, arg)


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
        ranking = self.rank()
        if not hasattr(self, "dump_file"):
            dump_file = os.path.curdir + "/TestWrapper/results.txt"
        else:
            dump_file = self.dump_file
        if os.path.isfile(dump_file):
            os.remove(dump_file)
        f = os.open(dump_file, os.O_WRONLY | os.O_CREAT)
        os.write(f, str(ranking).encode("utf-8"))
        os.close(f)


debugger = ReportingDebugger(collector_class=ExtendedCoverageCollector)
