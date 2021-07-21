from abc import abstractmethod
from typing import Any, Tuple, Set, Iterable


class RankerEvent:
    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int):
        self.event_type = str(self.__class__.__name__)
        self.program_element = program_element
        self.location = location
        self.passed_with_event = passed_with_event
        self.failed_with_event = failed_with_event
        self.total_passed = total_passed
        self.total_failed = total_failed

    @abstractmethod
    def __hash__(self):
        pass


class LineCoveredEvent(RankerEvent):
    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int):
        super(LineCoveredEvent, self).__init__(program_element, location, passed_with_event,
                                               failed_with_event, total_passed, total_failed)

    def __hash__(self):
        return hash((self.event_type, self.location))

    def __str__(self):
        return f"Covered {self.location}"


class SDBranchEvent(RankerEvent):
    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 outcome: bool):
        super(SDBranchEvent, self).__init__(program_element, location, passed_with_event,
                                            failed_with_event, total_passed, total_failed)
        self.outcome = outcome

    def __hash__(self):
        return hash((self.event_type, self.location, self.outcome))

    def __str__(self):
        return f"Branch @ {self.location} - {self.outcome}"


class SDReturnValueEvent(RankerEvent):
    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 operator: Any, outcome: bool):
        super(SDReturnValueEvent, self).__init__(program_element, location, passed_with_event,
                                                 failed_with_event, total_passed, total_failed)
        self.operator = operator
        self.outcome = outcome

    def __hash__(self):
        return hash((self.event_type, self.location, self.operator, self.outcome))

    def __str__(self):
        return f"Return v. {self.operator} - {self.outcome} @ {self.location}"


class SDScalarPairEvent(RankerEvent):
    def __init__(self, program_element: Any, location: Any, passed_with_event: Set[Any],
                 failed_with_event: Set[Any], total_passed: int, total_failed: int,
                 operator: Any, outcome: bool, operands: Tuple[str, str]):
        super(SDScalarPairEvent, self).__init__(program_element, location, passed_with_event,
                                                failed_with_event, total_passed, total_failed)
        self.operator = operator
        self.outcome = outcome
        self.operands = operands

    def __hash__(self):
        return hash((self.event_type, self.location, self.operands, self.operator, self.outcome))

    def __str__(self):
        return f"Pair {self.operands[0]} {self.operator} {self.operands[1]} - {self.outcome} @ {self.location}"


class EventContainer(Iterable):
    def __init__(self):
        self.events = dict()
        self.events_by_program_element = dict()

    def add(self, event: RankerEvent):
        h = hash(event)
        if h in self.events.keys():
            self.events[h].passed_with_event.update(event.passed_with_event)
            self.events[h].failed_with_event.update(event.failed_with_event)
        else:
            self.events[h] = event
            if event.program_element not in self.events_by_program_element:
                self.events_by_program_element[event.program_element] = {event}
            else:
                self.events_by_program_element[event.program_element].add(event)

    def __contains__(self, item):
        return hash(item) in self.events.keys()

    def __iter__(self):
        return iter(self.events.values())

    def get_from_program_element(self, program_element) -> Set[RankerEvent]:
        if program_element not in self.events_by_program_element.keys():
            return set()
        return self.events_by_program_element[program_element]
