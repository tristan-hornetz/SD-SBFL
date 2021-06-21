from abc import abstractmethod


class Predicate:
    """
    Abstract Base class for predicates
    """

    def __init__(self, _results):
        self.results = _results
        self.predicate_instances = dict()
        self.total_passed = len(self.results.collectors[self.results.PASS])
        self.total_failed = len(self.results.collectors[self.results.FAIL])
        for event in self.results.results:
            self.transform(event)

    def description(self) -> str:
        """
        :return: A high level description of what the predicate is
        """
        return self.__doc__

    @abstractmethod
    def transform(self, event) -> None:
        """
        Extract all info related to our predicate from a recorded event and adjust own data structures accordingly
        :param event: The event to be transformed
        """
        pass


class NoPredicate(Predicate):
    """
    Just rank events as they are recorded
    p(*) = True
    """
    def transform(self, event) -> None:
        passed_collectors = self.results.collectors_with_event[self.results.PASS][event]
        failed_collectors = self.results.collectors_with_event[self.results.FAIL][event]
        self.predicate_instances[event] = {k: {True, n} for k, n in passed_collectors}, \
                                          {k: {True, n} for k, n in failed_collectors}


class LineCoveredPredicate(Predicate):
    """
    p(*) = Line covered?
    """

    def transform(self, event) -> None:
        filename, method_name, lineno, event_type, *other = event
        if event_type == "Covered":
            passed_collectors = self.results.collectors_with_event[self.results.PASS][event]
            failed_collectors = self.results.collectors_with_event[self.results.FAIL][event]
            self.predicate_instances[(filename, method_name, lineno, "LineCoveredPredicate", *other)] = {k: {True, n} for k, n in passed_collectors}, \
                                              {k: {True, n} for k, n in failed_collectors}

