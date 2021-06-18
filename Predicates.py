from abc import abstractmethod
from typing import Tuple, Set, Dict


class Predicate:
    """
    Abstract Base class for predicates
    """

    def __init__(self, _results):
        self.results = _results

    def description(self) -> str:
        """
        :return: A high level description of what the predicate is
        """
        return self.__doc__

    @abstractmethod
    def check(self, event: Tuple[str, int], passed_ids: Set[int], failed_ids: Set[int],
              new_passed: Dict[str, Dict[int, bool]], new_failed: Dict[str, Dict[int, bool]]) -> None:
        """
        Check if the event is relevant to the predicate and adjust results accordingly
        :param event: An event recorded during a test run
        :param passed_ids: The ids of all passing tests where the event occurred
        :param failed_ids: The ids of all failing tests where the event occurred
        :param new_passed: Predicate @ Location -> (Collector ID -> Predicate Result) where the test passed
        :param new_failed: Predicate @ Location -> (Collector ID -> Predicate Result) where the test failed
        :return: None
        """
        pass


class NoPredicate(Predicate):
    """
    Just rank events as they are recorded
    p(*) = True
    """

    def check(self, event: Tuple[str, int], passed_ids: Set[int], failed_ids: Set[int],
              new_passed: Dict[str, Dict[int, bool]], new_failed: Dict[str, Dict[int, bool]]) -> None:

        e, lineno = event
        new_passed[f"{e}:{lineno}"] = {i: True for i in passed_ids}
        new_failed[f"{e}:{lineno}"] = {i: True for i in failed_ids}


class LineCoveredPredicate(NoPredicate):
    """
    p(*) = Line covered?
    """

    def check(self, event: Tuple[str, int], passed_ids: Set[int], failed_ids: Set[int],
              new_passed: Dict[str, Dict[int, bool]], new_failed: Dict[str, Dict[int, bool]]) -> None:
        if event[0].startswith("Covered "):
            super().check(event, passed_ids, failed_ids, new_passed, new_failed)

