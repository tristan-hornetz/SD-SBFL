from abc import abstractmethod
from typing import Dict, Tuple

from CodeInspection.Methods import DebuggerMethod
from RankerEvent import EventContainer, LineCoveredEvent, SDReturnValueEvent


class EventTranslator:
    @staticmethod
    @abstractmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        pass


class LineCoveredEventTranslator(EventTranslator):
    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        for event in filter(lambda e: e[3] == "Covered", _results.results):
            filename, method_name, lineno, *other = event
            passed_collectors = _results.collectors_with_event[_results.PASS][event]
            failed_collectors = _results.collectors_with_event[_results.FAIL][event]

            event_object = LineCoveredEvent(method_objects[filename, method_name, lineno],
                                            (filename, method_name, lineno), passed_collectors, failed_collectors,
                                            total_passed, total_failed)

            event_container.add(event_object)


class SDReturnValueEventTranslator(EventTranslator):
    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        comparable_types = set(str(t) for t in [int, str, float, bool])
        for event in filter(lambda e: e[3] == "Return", _results.results):
            filename, method_name, lineno, e_type, value, v_type = event
            if v_type in comparable_types:
                if v_type == str(str):
                    value = len(value)
                comparisons = [
                    ("==", value == 0),
                    ("!=", value != 0),
                    ("<", value < 0),
                    ("<=", value <= 0),
                    (">", value > 0),
                    (">=", value >= 0)
                ]
                passed_collectors = _results.collectors_with_event[_results.PASS][event]
                failed_collectors = _results.collectors_with_event[_results.FAIL][event]
                for op, result in comparisons:
                    event_object = SDReturnValueEvent(method_objects[filename, method_name, lineno],
                                                      (filename, method_name, lineno), passed_collectors,
                                                      failed_collectors,
                                                      total_passed, total_failed, op, result)
                    event_container.add(event_object)
