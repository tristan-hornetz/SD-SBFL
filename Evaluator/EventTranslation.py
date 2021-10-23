from typing import Dict, Optional

from .CodeInspection.Branches import extractBranchesFromCode
from .CodeInspection.Methods import extractMethodsFromCode, BugInfo, DebuggerMethod
from .RankerEvent import *


class EventTranslator:
    """
    EventTranslators are utilized to translate events from their recorded from to the form utilized
    by the Evaluation Framework
    """

    @staticmethod
    def match_method(filename, method_name, lineno, method_objects):
        if str(method_name).startswith("<") and method_name != "<module>":
            possible_methods = list(
                filter(lambda m: m[0] == filename and lineno == m[2] and "<" not in m[1], method_objects.keys()))
            if len(possible_methods) > 0:
                method_name = possible_methods[0][1]
        return filename, method_name, lineno

    @staticmethod
    @abstractmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        """
        Translate results from the Recording Framework to a form usable by the Evaluation Framework

        :param _results: The output of the Recording Framework
        :param event_container: The EventContainer instance to add the newly generated events to
        :param method_objects: The program's methods as extracted from code
        """
        pass


class LineCoveredEventTranslator(EventTranslator):
    """
    Translates recorded events into LineCoveredEvents
    """

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        for event in filter(lambda e: e[3] == "Covered", _results.results):
            filename, method_name, lineno, *other = event
            filename, method_name, lineno = EventTranslator.match_method(filename, method_name, lineno, method_objects)
            if (filename, method_name, lineno) not in method_objects.keys():
                continue
            passed_collectors = _results.collectors_with_event[_results.PASS][event]
            failed_collectors = _results.collectors_with_event[_results.FAIL][event]

            event_object = LineCoveredEvent(method_objects[filename, method_name, lineno],
                                            (filename, method_name, lineno), passed_collectors, failed_collectors,
                                            total_passed, total_failed)

            event_container.add(event_object)


class SDBranchEventTranslator(EventTranslator):
    """
    Translates recorded events into SDBranchEvents
    """

    @staticmethod
    def collectors_for_event(_results, event):
        collectors = dict()
        if event in _results.collectors_with_event[_results.FAIL].keys():
            collectors = _results.collectors_with_event[_results.FAIL][event].copy()
        if event in _results.collectors_with_event[_results.PASS].keys():
            collectors.update(_results.collectors_with_event[_results.PASS][event])
        return {c[0]: c[1] for c in collectors}

    @staticmethod
    def create_event(_results, event_container, branch, collectors_for, outcome):
        if len(collectors_for[outcome]) == 0:
            return
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        event_container.add(SDBranchEvent(branch.method_object,
                                          branch.location,
                                          collectors_for[outcome].intersection(_results.collectors[_results.PASS]),
                                          collectors_for[outcome].intersection(_results.collectors[_results.FAIL]),
                                          total_passed, total_failed, outcome))

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        info = BugInfo(_results)
        branches = extractBranchesFromCode(_results, info, method_objects)
        covered_locations = {
            (filename.replace(info.work_dir + "/", ""), method_name, lineno): (filename, method_name, lineno, *o) for
            filename, method_name, lineno, *o in filter(lambda e: e[3] == "Covered", _results.results)}
        for branch in branches:
            if branch.location not in covered_locations.keys():
                continue
            head_event = covered_locations[branch.location]
            head_in_collectors = SDBranchEventTranslator.collectors_for_event(_results, head_event)
            body_in_collectors = dict()
            if (branch.location[0], branch.location[1], branch.first_body_lineno) in covered_locations.keys():
                body_event = covered_locations[(branch.location[0], branch.location[1], branch.first_body_lineno)]
                body_in_collectors = SDBranchEventTranslator.collectors_for_event(_results, body_event)
            combined_dict = {c: (head_in_collectors[c], body_in_collectors[c] if c in body_in_collectors.keys() else 0)
                             for c in head_in_collectors.keys()}
            collectors_for = {True: set(), False: set()}

            for collector, (head, body) in combined_dict.items():
                if body > 0:
                    collectors_for[True].add(collector)
                if body < head:
                    collectors_for[False].add(collector)

            SDBranchEventTranslator.create_event(_results, event_container, branch, collectors_for, True)
            SDBranchEventTranslator.create_event(_results, event_container, branch, collectors_for, False)


class SDReturnValueEventTranslator(EventTranslator):
    """
    Translates recorded events into SDReturnValueEvents
    """

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        comparable_types = set(str(t) for t in [int, str, float, bool])
        for event in filter(lambda e: e[3] == "Return", _results.results):
            filename, method_name, lineno, e_type, value, v_type = event
            if (filename, method_name, lineno) not in method_objects.keys():
                continue
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


class SDScalarPairEventTranslator(EventTranslator):
    """
    Translates recorded events into SDScalarPair
    """

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        for event in filter(lambda e: e[3] == "Scalar" or e[3] == "Pair", _results.results):
            filename, method_name, lineno, e_type, *other = event
            filename, method_name, lineno = EventTranslator.match_method(filename, method_name, lineno, method_objects)
            if (filename, method_name, lineno) not in method_objects.keys():
                continue
            passed_collectors = _results.collectors_with_event[_results.PASS][event]
            failed_collectors = _results.collectors_with_event[_results.FAIL][event]
            if e_type == "Scalar":
                (name, value), *other = other
                o_name = "<0>"
                comparisons = [
                    ("==", value == 0),
                    ("!=", value != 0),
                    ("<", value < 0),
                    ("<=", value <= 0),
                    (">", value > 0),
                    (">=", value >= 0)
                ]
            else:
                (name, o_name), (equal, smaller) = other
                comparisons = [
                    ("==", equal),
                    ("!=", not equal),
                    ("<", smaller),
                    ("<=", smaller or equal),
                    (">", not smaller and not equal),
                    (">=", not smaller)
                ]

            for op, result in comparisons:
                event_object = SDScalarPairEvent(method_objects[filename, method_name, lineno],
                                                 (filename, method_name, lineno), passed_collectors,
                                                 failed_collectors, total_passed, total_failed, op, result,
                                                 (name, o_name))
                event_container.add(event_object)


class AbsoluteReturnValueEventTranslator(EventTranslator):
    """
    Translates recorded events into AbsoluteReturnValueEvents
    """

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        for event in filter(lambda e: e[3] == "Return", _results.results):
            filename, method_name, lineno, _, value, type_str = event
            filename, method_name, lineno = EventTranslator.match_method(filename, method_name, lineno, method_objects)
            if (filename, method_name, lineno) not in method_objects.keys():
                continue
            passed_collectors = _results.collectors_with_event[_results.PASS][event]
            failed_collectors = _results.collectors_with_event[_results.FAIL][event]

            event_object = AbsoluteReturnValueEvent(method_objects[filename, method_name, lineno],
                                                    (filename, method_name, lineno), passed_collectors,
                                                    failed_collectors, total_passed, total_failed, value)

            event_container.add(event_object)


class AbsoluteScalarValueEventTranslator(EventTranslator):
    """
    Translates recorded events into AbsoluteScalarValueEvents
    """

    @staticmethod
    def translate(_results, event_container: EventContainer,
                  method_objects: Dict[Tuple[str, str, int], DebuggerMethod]):
        total_passed = len(_results.collectors[_results.PASS])
        total_failed = len(_results.collectors[_results.FAIL])
        for event in filter(lambda e: e[3] == "Scalar", _results.results):
            filename, method_name, lineno, _, (var_name, value) = event
            filename, method_name, lineno = EventTranslator.match_method(filename, method_name, lineno, method_objects)
            if (filename, method_name, lineno) not in method_objects.keys():
                continue
            passed_collectors = _results.collectors_with_event[_results.PASS][event]
            failed_collectors = _results.collectors_with_event[_results.FAIL][event]

            event_object = AbsoluteScalarValueEvent(method_objects[filename, method_name, lineno],
                                                    (filename, method_name, lineno), passed_collectors,
                                                    failed_collectors, total_passed, total_failed, var_name, value)

            event_container.add(event_object)


DEFAULT_TRANSLATORS = [LineCoveredEventTranslator, SDBranchEventTranslator, SDScalarPairEventTranslator,
                       SDReturnValueEventTranslator, AbsoluteReturnValueEventTranslator,
                       AbsoluteScalarValueEventTranslator]


class EventProcessor:
    """
    Utilized to translate the recorded events using multiple EventTranslators
    """

    def __init__(self, translators: Iterable[Optional[EventTranslator, type]]):
        """
        :param translators: The event translators to utilize
        """
        self.translators = translators

    def process(self, _results) -> Tuple[EventContainer, Dict[Tuple[str, str, int], DebuggerMethod], BugInfo]:
        """
        Translate the events in _results

        :param _results: The output of the Recording Framework
        :return: The output of the translation, consisting of an EventContainer instance, the extracted methods and a BugInfo object
        """
        event_container = EventContainer()
        info = BugInfo(_results)
        # print("Extracting method objects")
        method_objects = extractMethodsFromCode(_results, info)
        if len(method_objects.items()) < 1:
            print(f"{_results.project_name}, {_results.bug_id} - No methods extracted")
            raise AssertionError()
        for t in self.translators:
            # print(f"Translating for {t}")
            t.translate(_results, event_container, method_objects)
        return event_container, method_objects, info
