import math
from abc import abstractmethod
from typing import Tuple, SupportsFloat

from .RankerEvent import RankerEvent


class SimilarityCoefficient:
    """
    Base class for similarity coefficients
    """

    @staticmethod
    def get_values(event: RankerEvent) -> Tuple[int, int, int, int, int, int]:
        """
        Extract all required values from the given event

        :param event: The event to extract the values from
        :return: A tuple of required values
        """
        failed = len(event.failed_with_event)
        passed = len(event.passed_with_event)
        total_failed = max(event.total_failed, failed)
        total_passed = event.total_passed
        not_in_failed = total_failed - failed
        not_in_passed = total_passed - passed
        return passed, failed, not_in_passed, not_in_failed, total_passed, total_failed

    @staticmethod
    @abstractmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        """
        Compute the similarity coefficient of the given event

        :param event: The event to compute the similarity coefficient for
        :return: The event's similarity coefficient
        """
        pass


class OchiaiCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return failed / math.sqrt(total_failed * (failed + passed))
        except ZeroDivisionError:
            return 0.0


class JaccardCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return failed / (total_failed + passed)
        except ZeroDivisionError:
            return 0.0


class SorensenDiceCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return (2.0 * failed) / ((2.0 * failed) + not_in_failed + passed)
        except ZeroDivisionError:
            return 0.0


class AnderbergCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return failed / (failed + 2 * (not_in_failed + passed))
        except ZeroDivisionError:
            return 0.0


class SimpleMatchingCoefficient(SimilarityCoefficient):
    @staticmethod
    @abstractmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return (failed + not_in_passed) / (total_failed + event.total_passed)
        except ZeroDivisionError:
            return 0.0


class RogersTanimotoCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return (failed + not_in_passed) / (
                failed + not_in_passed + 2 * (not_in_failed + passed)
            )
        except ZeroDivisionError:
            return 0.0


class OchiaiIICoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return (failed * not_in_passed) / math.sqrt(
                total_failed
                * (failed + passed)
                * (not_in_failed + not_in_passed)
                * total_passed
            )
        except ZeroDivisionError:
            return 0.0
        except ValueError:
            return 0.0


class RusselRaoCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return failed / (total_failed + total_passed)
        except ZeroDivisionError:
            return 0.0


class TarantulaCoefficient(SimilarityCoefficient):
    @staticmethod
    def compute(event: RankerEvent) -> SupportsFloat:
        (
            passed,
            failed,
            not_in_passed,
            not_in_failed,
            total_passed,
            total_failed,
        ) = SimilarityCoefficient.get_values(event)
        try:
            return (failed / total_failed) / (
                (passed / total_passed) + (failed / total_failed)
            )
        except ZeroDivisionError:
            return 0.0
