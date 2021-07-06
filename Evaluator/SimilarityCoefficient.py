import math
from abc import abstractmethod
from .RankerEvent import RankerEvent


class SimilarityCoefficient:
    @staticmethod
    @abstractmethod
    def compute(event: RankerEvent):
        pass


class OchiaiCoefficient(SimilarityCoefficient):
    @staticmethod
    @abstractmethod
    def compute(event: RankerEvent):
        failed = len(event.failed_with_event)
        passed = len(event.passed_with_event)
        total_failed = event.total_failed
        try:
            return failed / math.sqrt(total_failed * (failed + passed))
        except ZeroDivisionError:
            return 0.0
